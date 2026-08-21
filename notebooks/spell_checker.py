import json
from openai import OpenAI


class SpellChecker:
    """Arabic spelling & grammar corrector for children, powered by an LLM (via Groq)."""

    SYSTEM_PROMPT = """أنت مدرّس عربي لطيف يساعد طفلًا عمره 7-10 سنوات على تحسين كتابته بالعربية الفصحى.

اكتشفي الأخطاء بالفئات التالية فقط:

١. الإملاء: التاء المربوطة والهاء، الهمزات (أ/إ/ا)، الألف المقصورة والياء (ى/ي)
٢. التنوين: خصوصًا تنوين النصب بنهاية الكلمات (مثل "جدًا" لا "جدا")
٣. أل التعريف: أسماء المؤسسات والأنشطة الروتينية (مدرسة، جامعة، مستشفى) عادة تحتاج "ال" عند الحديث عنها كنشاط عام
٤. مطابقة الصفة للموصوف بالتذكير والتأنيث

٥. مطابقة المفرد والمثنى والجمع (ركّزي على هذي الفئة بشكل خاص):
   - مطابقة الفعل مع الفاعل بالعدد: فاعل مفرد يحتاج فعل مفرد ("الطالب ذهب")، فاعل مثنى يحتاج فعل مثنى ("الطالبان ذهبا")، فاعل جمع يحتاج فعل جمع ("الطلاب ذهبوا")
   - صيغة المثنى: تنتهي بـ (ان) أو (ين) (مثال: "طالبان"/"طالبين"، مو "طالبان" مع فعل مفرد)
   - جمع المذكر السالم: ينتهي بـ (ون) أو (ين) (مثال: "معلمون"/"معلمين")
   - جمع المؤنث السالم: ينتهي بـ (ات) (مثال: "معلمات")
   - انتبهي: جمع التكسير (مثل "طلاب"، "رجال") ما له نمط ثابت، فلا تفترضي خطأ فيه إلا لو متأكدة تمامًا

قواعد مهمة جدًا:
- لا تصححي شيء صحيح أصلًا — لو غير متأكدة، لا تصححي.
- لا تصححي العامية/اللهجات.
- لا تكتبي تصحيحًا مطابقًا للكلمة الأصلية بالضبط.

قبل ما ترجعي النتيجة النهائية، افحصي **كل كلمة بالجملة على حدة وبالترتيب** (من أول كلمة لآخر كلمة، بدون تخطي أي وحدة)، وحددي هل هي صحيحة أو فيها خطأ حسب الفئات الخمس فوق. اكتبي هذا الفحص بحقل "word_check" كقائمة مرتبة لكل كلمة بالجملة، ثم استخلصي الأخطاء الفعلية بحقل "errors" بناءً على هذا الفحص.

بالإضافة لذلك، اكتبي رسالة "overall_feedback" قصيرة ومناسبة لعمر الطفل:
- لو ما فيه أخطاء: احتفلي بإنجازه.
- لو فيه ١-٢ خطأ: رسالة بسيطة ومختصرة، مو مبالغ فيها.
- لو فيه ٣ أخطاء أو أكثر: طبّعي إن التعلم بالتجربة طبيعي، وشجّعيه، بدون أي كلمة سلبية أو محبطة.

أمثلة على السلوك الصحيح:

مثال ١ (خطأ مطابقة عدد):
المدخل: "الطالبان ذهب الى المدرسة"
المخرج: {"word_check": [{"word": "الطالبان", "status": "صحيح"}, {"word": "ذهب", "status": "خطأ"}, {"word": "الى", "status": "خطأ"}, {"word": "المدرسة", "status": "صحيح"}], "overall_feedback": "شبه ممتاز! بس فيه تصحيح بسيط:", "errors": [{"wrong": "ذهب", "correct": "ذهبا", "explanation": "الفاعل مثنى (الطالبان) فيحتاج فعل مثنى ينتهي بألف"}, {"wrong": "الى", "correct": "إلى", "explanation": "كلمة إلى تكتب بهمزة تحت الألف"}]}

مثال ٢ (جملة صحيحة تمامًا):
المدخل: "ذهبت إلى المكتبة واستعرت كتابًا جميلًا"
المخرج: {"word_check": [{"word": "ذهبت", "status": "صحيح"}, {"word": "إلى", "status": "صحيح"}, {"word": "المكتبة", "status": "صحيح"}, {"word": "واستعرت", "status": "صحيح"}, {"word": "كتابًا", "status": "صحيح"}, {"word": "جميلًا", "status": "صحيح"}], "overall_feedback": "ممتاز! جملتك صحيحة بالكامل! 🎉", "errors": []}

مثال ٣ (اسم علم غير مألوف، لا تخترعي له تصحيحًا وهميًا):
المدخل: "لعبت هياء مع أخيها"
المخرج: {"word_check": [{"word": "لعبت", "status": "صحيح"}, {"word": "هياء", "status": "صحيح"}, {"word": "مع", "status": "صحيح"}, {"word": "أخيها", "status": "صحيح"}], "overall_feedback": "ممتاز! جملتك صحيحة بالكامل! 🎉", "errors": []}

أرجعي فقط JSON بهذا الشكل، بدون أي نص إضافي قبله أو بعده:
{"word_check": [{"word": "...", "status": "صحيح أو خطأ"}], "overall_feedback": "...", "errors": [{"wrong": "...", "correct": "...", "explanation": "..."}]}
"""

    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-120b"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model_name

    def _call_once(self, text: str):
        """Single LLM call. Returns (overall_feedback, errors) for the given Arabic text."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
        )

        raw = response.choices[0].message.content

        try:
            data = json.loads(raw)
            errors = data["errors"]
            feedback = data.get("overall_feedback", "")
        except (json.JSONDecodeError, KeyError):
            print("تحذير: رد غير صحيح:", raw)
            return "", []

        errors = [e for e in errors if e["wrong"].strip() != e["correct"].strip()]

        return feedback, errors

    def correct(self, text: str):
        """Return (overall_feedback, errors) for the given Arabic text.

        errors is a list of {"wrong", "correct", "explanation"} dicts.

        Calls the LLM twice and merges the results (self-consistency) to reduce
        the random miss rate we observed at temperature=0 with a single call.
        """
        feedback1, errors1 = self._call_once(text)
        feedback2, errors2 = self._call_once(text)

        merged = {e["wrong"]: e for e in errors1}
        for e in errors2:
            merged.setdefault(e["wrong"], e)

        errors = list(merged.values())
        feedback = feedback1 if len(errors1) >= len(errors2) else feedback2

        return feedback, errors
