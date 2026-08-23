import re
import time
import random


class AneesWordGame:

    def __init__(self, client, age_range="6 إلى 11 سنة", debug=False):
        self.client = client
        self.age_range = age_range
        self.debug = debug

    def ask_groq(self, prompt, temperature=0.3, reasoning_effort="low", retries=3):
        """
        إذا فشل الاتصال بـ Groq بسبب مشكلة مؤقتة، نحاول إرسال الطلب مرة ثانية.
        وإذا فشلت كل المحاولات، نرجع None بدل ما يتوقف البرنامج.
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    reasoning_effort=reasoning_effort
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if self.debug:
                    print(f"[خطأ API] محاولة {attempt+1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"[فشل نهائي] تعذر الاتصال بعد {retries} محاولات: {e}")
                    return None

    _KNOWN_GENDER_FIXES = {
        "ما هي الشعور": "ما هو الشعور",
        "ما هو الظاهرة": "ما هي الظاهرة",
        "ما هي الشيء": "ما هو الشيء",
        "ما هو الحركة": "ما هي الحركة",
        "ما هو الصفة": "ما هي الصفة",
        "ما هو الكلمة": "ما هي الكلمة",
        "ما هي المكان": "ما هو المكان",
        "ما هي الصوت": "ما هو الصوت",
        "ما هي الجزء": "ما هو الجزء",
        "ما هي الوعاء": "ما هو الوعاء",
        "ما هو الأداة": "ما هي الأداة",
    }

    def _fix_gender_agreement(self, question):
        """
        بدل ما نرسل طلب جديد لـ Groq لتصحيح صيغة السؤال، استخدمنا قاموس
        يحتوي على أكثر الأخطاء التي تتكرر معنا، وهذا يقلل استهلاك الـ API.
        """
        for wrong, right in self._KNOWN_GENDER_FIXES.items():
            question = question.replace(wrong, right)
        return question

    @staticmethod
    def _normalize(text):
        text = text.replace('ال', '', 1) if text.startswith('ال') else text
        diacritics = 'ًٌٍَُِّْ'
        return ''.join(c for c in text if c not in diacritics).strip()

    @staticmethod
    def _format_options(options):
        """
        نرتب الخيارات بشكل مرقم وواضح حتى يقدر نظام التحقق يميز كل خيار برقم.
        """
        return "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))

    def _verify_and_get_rejected(self, word, meaning, question, options):
        """
        نجمع التحقق من الخيارات وتحديد الخيارات المرفوضة في دالة واحدة
        بدل ما نكرر نفس الخطوتين أكثر من مرة.
        """
        verification_result = self.verify_options(word, meaning, question, options)
        rejected_indexes = self.get_rejected_indexes(verification_result)
        return verification_result, rejected_indexes

    COMMON_QUESTION_RULES = """
قواعد مهمة:
- يجب أن تكون بداية السؤال صحيحة نحويًا ("ما هي" للمؤنث، "ما هو" للمذكر).
- استخدم عبارة أو فئة عامة تصف نوع الكلمة، وتكون مختلفة عن الكلمة الصحيحة.
- صف مشهدًا أو موقفًا ملموسًا يستطيع الطفل تخيله بسهولة.
- لا تذكر الكلمة الصحيحة أو أي جزء منها في السؤال إطلاقًا، ولا داخل مثال أو وصف.
- يجب أن يحافظ السؤال على نوع الكلمة النحوي (اسم/فعل/صفة) كما ورد بالكلمة نفسها.
- لا تستنتج نوع الكلمة من طبيعة المعنى المرجعي، اعتمد على الكلمة نفسها.
- لا توسّع المعنى المرجعي إلى معنى آخر أو مجازي غير موجود فيه.
- إذا كانت الكلمة اسمًا لشيء، اجعل السؤال عن الشيء نفسه، وليس عن استخدامه أو أثره.
- استخدم الخاصية الأكثر تمييزًا بالمعنى المرجعي، تجنب الوصف العام جدًا.
- اجعل الوصف كافيًا لتمييز الكلمة الصحيحة عن كلمات أخرى بنفس المجال.
- استخدم لغة عربية فصحى بسيطة مناسبة لعمر 6 إلى 11 سنة.
- اجعل السؤال واضحًا وله إجابة صحيحة واحدة فقط.
- لا تكتب خيارات، لا تكتب شرحًا، لا تستخدم <think>.
- أرجع السؤال فقط.
"""

    SELF_CHECK = """
قبل إخراج السؤال، تحقق من:
1. هل السؤال يطلب نفس نوع الكلمة؟
2. هل يعتمد على المعنى المرجعي فقط؟
3. هل توجد كلمة أخرى ممكن تكون إجابة صحيحة بدل الكلمة المطلوبة؟
إذا كانت الإجابة "لا" عن أي منها، أعد الصياغة قبل الإخراج.
"""

    def _build_question_prompt(self, word, meaning, emphasize_no_leak=False):
        emphasis = (
            f'\nتذكير حرج: يجب ألا تظهر الكلمة "{word}" ولا أي جزء منها '
            f'بالسؤال إطلاقًا، حتى لو احتجت تستخدم فئة أعم أو وصفًا غير مباشر.'
            if emphasize_no_leak else ""
        )
        return f"""
أنت مساعد تعليمي للأطفال في منصة أنيس.

الفئة العمرية المستهدفة: {self.age_range}

الكلمة:
{word}

المعنى المرجعي:
{meaning}

المهمة:
حوّل المعنى المرجعي إلى سؤال قصير وواضح يستطيع الطفل معرفة الكلمة
الصحيحة من خلاله. لا تجعل صيغة السؤال ثابتة لكل الكلمات؛ اختر صياغة
تناسب نوع الكلمة ووظيفتها ومعناها.
{self.COMMON_QUESTION_RULES}
{self.SELF_CHECK}
{emphasis}
"""

    def generate_question(self, word, meaning, max_attempts=2):
        """
        نحاول توليد السؤال أكثر من مرة، وإذا ظهرت الكلمة الصحيحة داخل السؤال
        نحاول مرة ثانية مع تشديد التعليمات لمنع ظهورها.
        """
        clean_word = self._normalize(word)

        for attempt in range(max_attempts):
            prompt = self._build_question_prompt(word, meaning, emphasize_no_leak=False)
            question = self.ask_groq(prompt, temperature=0.3 + attempt * 0.1)

            if question is None:
                continue

            question = self._fix_gender_agreement(question.strip())

            if clean_word not in self._normalize(question):
                if self.debug:
                    print(f"[محاولة {attempt+1}] نجح: {question}")
                return question

            if self.debug:
                print(f"[محاولة {attempt+1}] سرّب الكلمة: {question}")

        prompt = self._build_question_prompt(word, meaning, emphasize_no_leak=True)
        question = self.ask_groq(prompt, temperature=0.5)

        if question is None:
            if self.debug:
                print(f"[فشل كامل] ما قدرنا نولّد سؤال لـ '{word}' — مشكلة اتصال")
            return None

        question = self._fix_gender_agreement(question.strip())

        if self.debug:
            status = "نجح" if clean_word not in self._normalize(question) else "لسا مسرب"
            print(f"[محاولة احتياطية] {status}: {question}")

        return question

    def generate_distractors(self, word, meaning, question, max_attempts=2):
        prompt = f"""
أنت مساعد تعليمي في لعبة الكلمات في منصة أنيس.

الفئة العمرية المستهدفة: {self.age_range}

الكلمة الصحيحة:
{word}

المعنى المرجعي:
{meaning}

السؤال:
{question}

المهمة:
أنشئ 3 كلمات مشتتة مناسبة للسؤال، تبدو معقولة للطفل لكن يجب أن تكون
هناك إجابة صحيحة واحدة فقط.

القواعد الأساسية:
1. كل كلمة مشتتة مختلفة بوضوح في المعنى عن الكلمة الصحيحة.
2. لا تستخدم مرادفًا، شبه مرادف، كلمة يمكن أن تحل محل الكلمة الصحيحة،
   أو كلمة من نفس الجذر أو مشتقة منها.
3. يمكن أن تكون الكلمة مرتبطة بالسياق العام للسؤال، بشرط معناها مختلف
   بوضوح عن الكلمة الصحيحة.
4. لا تجعل المشتتات بعيدة جدًا أو عشوائية بلا علاقة بالسياق.
5. الأولوية أن تكون كل كلمة مختلفة بوضوح عن الكلمة الصحيحة، لا أن
   تتشابه الثلاث كلمات مع بعضها.
6. استخدم نفس النوع النحوي للكلمة الصحيحة (اسم مع اسم، فعل مع فعل...).
7. كل كلمة مناسبة لطفل من عمر 6 إلى 11 سنة، عربية صحيحة ومستخدمة طبيعيًا.
8. لا كلمات مصطنعة، نادرة جدًا، ناقصة، أو مركبة بشكل غير طبيعي.
9. مهم جدًا: كل مشتت يجب أن يكون كلمة واحدة مستقلة بدون مسافات إطلاقًا
   (ممنوع أي عبارة من كلمتين أو أكثر مثل "ضحكة الأطفال" أو "صوت المطر").

مثال:
الكلمة الصحيحة: تعاون
لا تستخدم (قريبة جدًا): مساعدة، مشاركة، شارك، تفاعل
استخدم بدلها (مرتبطة بالسياق لكن مختلفة المعنى): مسابقة، رحلة، مغامرة

قبل الإخراج، اسأل نفسك عن كل خيار: هل يمكن اعتباره إجابة صحيحة؟ هل هو
مرادف أو قريب جدًا من المعنى؟ هل هو أكثر من كلمة واحدة؟ لو الإجابة
"نعم" على أي منها، استبدله.

- أرجع 3 كلمات فقط، كل كلمة بسطر مستقل، كل كلمة مفردة بدون مسافات.
- لا أرقام، لا شرح، لا <think>.
"""
        single_word = []
        for attempt in range(max_attempts):
            response = self.ask_groq(prompt, temperature=0.4 + attempt * 0.02)

            if response is None:
                continue

            distractors = [line.strip() for line in response.splitlines() if line.strip()]
            single_word = [d for d in distractors if len(d.split()) == 1]

            if len(single_word) >= 3:
                return single_word[:3]

            if self.debug:
                rejected = [d for d in distractors if len(d.split()) > 1]
                print(f"[محاولة {attempt+1}] مشتتات مرفوضة (أكثر من كلمة): {rejected}")

        if self.debug:
            print(f"[تحذير نهائي] ما قدرنا نجمع 3 مشتتات كلمة واحدة لـ '{word}'")

        return single_word[:3]

    def verify_options(self, word, meaning, question, options):
        options_text = self._format_options(options)
        prompt = f"""
أنت نظام تحقق للعبة الكلمات في منصة أنيس.

الفئة العمرية المستهدفة: {self.age_range}

الكلمة الصحيحة:
{word}

المعنى المرجعي:
{meaning}

السؤال:
{question}

الخيارات:
{options_text}

مهمتك: تحقق من الخيارات الأربعة.

القواعد:
1. الخيار 0 هو الكلمة الصحيحة، PASS دائمًا.
2. الخيار الآخر REJECT فقط إذا كان: إجابة صحيحة بديلة فعلية لنفس السؤال،
   أو مرادفًا واضحًا ومباشرًا، أو نفس المعنى الأساسي بدرجة يتعذر التمييز.
3. لا ترفض لمجرد أنه من نفس المجال، مرتبط بالموقف/الموضوع، أو يمثل
   شعورًا/شيئًا مشابهًا لكنه ليس نفس الشيء بالضبط.

أمثلة:
الكلمة "خجل": "خوف"=PASS، "توتر"=PASS، "انزعاج"=PASS (مشاعر مختلفة، مو مرادفات)
الكلمة "برق": "وميض"=PASS، "غيوم"=PASS، "رياح"=PASS (مرتبطة بالظاهرة، مو مرادفة)
الكلمة "بداية": "مستهل"=REJECT، "افتتاح"=REJECT (قريبة جدًا، بدائل معقولة)

كن متساهلًا مع الارتباط بالموضوع، ومتشددًا فقط مع الإجابة الصحيحة البديلة
أو المرادف الواضح.

لا شرح. أرجع فقط 4 أسطر بهذا الشكل بالضبط:
0: PASS
1: PASS أو REJECT
2: PASS أو REJECT
3: PASS أو REJECT
"""
        return self.ask_groq(prompt, temperature=0.0, reasoning_effort="low")

    def get_rejected_indexes(self, verification_result, expected_count=4):
        if verification_result is None:
            if self.debug:
                print("[تحذير] فشل التحقق (اتصال)، نرفض كل شي احترازيًا")
            return list(range(1, expected_count))

        rejected_indexes = []
        pattern = re.compile(r'^\s*(\d+)\s*:\s*(PASS|REJECT)\s*$')
        matched_lines = 0

        for line in verification_result.splitlines():
            match = pattern.match(line.strip())

            if match:
                matched_lines += 1
                index = int(match.group(1))
                status = match.group(2)

                if status == "REJECT" and index != 0:
                    rejected_indexes.append(index)

        if matched_lines < expected_count:
            if self.debug:
                print(f"[تحذير] التحقق رجّع {matched_lines}/{expected_count} أسطر — نرفض كل شي احترازيًا")

            rejected_indexes = list(range(1, expected_count))

        return rejected_indexes

    def generate_replacements(self, word, meaning, question, options, rejected_indexes):
        options_text = self._format_options(options)
        prompt = f"""
أنت مساعد تعليمي لمنصة أنيس للأطفال من عمر 6 إلى 11 سنة.

الكلمة الصحيحة:
{word}

المعنى المرجعي:
{meaning}

السؤال:
{question}

الخيارات الحالية:
{options_text}

أرقام الخيارات المرفوضة:
{rejected_indexes}

المهمة: اقترح كلمة بديلة لكل خيار مرفوض.

الشروط:
- معقولة بسياق السؤال، لكن مختلفة بوضوح عن الكلمة الصحيحة.
- ليست إجابة صحيحة، ليست مرادفًا، ليست مشتقة من الكلمة الصحيحة.
- لا تستخدم أي كلمة موجودة أصلًا بالخيارات، وكل كلمة مختلفة عن الأخرى.
- كلمة عربية صحيحة ومستخدمة طبيعيًا، مناسبة لعمر 6-11.
- لا كلمات مصطنعة، مقطوعة، أو مركّبة بشكل غير صحيح.

- أرجع عدد كلمات يساوي عدد الخيارات المرفوضة، كل كلمة بسطر مستقل.
- لا أرقام، لا شرح، لا <think>.
"""
        response = self.ask_groq(prompt, temperature=0.4)

        if response is None:
            return []

        replacements = [line.strip() for line in response.splitlines() if line.strip()]
        return replacements

    def generate_game(self, word, meaning, max_attempts=3):
        word = word.strip()
        meaning = meaning.strip()

        question = self.generate_question(word, meaning)

        if question is None:
            return {
                "success": False,
                "word": word,
                "meaning": meaning,
                "question": None,
                "options": [],
                "correct_index": None,
                "verification": None,
                "rejected_indexes": [],
                "attempts": 0,
                "error": "فشل الاتصال بالكامل بعد كل المحاولات",
            }

        distractors = self.generate_distractors(word, meaning, question)
        options = [word] + distractors[:3]

        verification_result, rejected_indexes = self._verify_and_get_rejected(
            word, meaning, question, options
        )

        attempt = 1

        if self.debug:
            print(f"[محاولة {attempt}] مرفوض: {rejected_indexes}")

        while rejected_indexes and attempt < max_attempts:
            attempt += 1

            replacements = self.generate_replacements(
                word, meaning, question, options, rejected_indexes
            )

            if len(replacements) < len(rejected_indexes) and self.debug:
                print(
                    f"[تحذير] طلبنا {len(rejected_indexes)} بدائل، "
                    f"رجع {len(replacements)}"
                )

            for index, replacement in zip(rejected_indexes, replacements):
                options[index] = replacement

            verification_result, rejected_indexes = self._verify_and_get_rejected(
                word, meaning, question, options
            )

            if self.debug:
                print(f"[محاولة {attempt}] مرفوض: {rejected_indexes}")

        seen = set()
        duplicate_indexes = []

        for index, option in enumerate(options):
            normalized_option = self._normalize(option)

            if normalized_option in seen:
                duplicate_indexes.append(index)
            else:
                seen.add(normalized_option)

        if duplicate_indexes:
            if self.debug:
                print(f"[تكرار] خيارات مكررة: {duplicate_indexes}")

            replacements = self.generate_replacements(
                word, meaning, question, options, duplicate_indexes
            )

            if len(replacements) < len(duplicate_indexes) and self.debug:
                print(
                    f"[تحذير] طلبنا {len(duplicate_indexes)} بدائل للتكرار، "
                    f"رجع {len(replacements)}"
                )

            for index, replacement in zip(duplicate_indexes, replacements):
                options[index] = replacement

            verification_result, rejected_indexes = self._verify_and_get_rejected(
                word, meaning, question, options
            )

        correct_word = word
        shuffled_options = options[:]

        random.shuffle(shuffled_options)

        correct_index = shuffled_options.index(correct_word)

        return {
            "success": len(rejected_indexes) == 0,
            "word": word,
            "meaning": meaning,
            "question": question,
            "options": shuffled_options,
            "correct_index": correct_index,
            "verification": verification_result,
            "rejected_indexes": rejected_indexes,
            "attempts": attempt,
        }