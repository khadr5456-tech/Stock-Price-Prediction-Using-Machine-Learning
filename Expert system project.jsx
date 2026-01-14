/* Expert System Project — Medical Diagnostic Single-file React component (Tailwind CSS) Default export: ExpertDiagnosisApp

Includes:

Project description & deliverables

Facts, Rules, Inference engine (forward chaining)

Sample rules and symptoms

Professional GUI using Tailwind and shadcn/ui imports example

Team roles, demo plan, grading mapping to 5+5+5


How to use:

Paste this file into a React app (create-react-app or Next.js page)

Tailwind should be available

shadcn/ui components optional; fallbacks included */


import React, { useMemo, useState } from "react"; // If you use shadcn/ui or other component libraries, you can replace simple elements below

const SYMPTOMS = [ "fever", "cough", "shortness_of_breath", "sore_throat", "runny_nose", "headache", "fatigue", "chest_pain", "loss_of_smell_or_taste", "nausea_or_vomiting", ];

// Human-readable labels const SYMPTOM_LABEL = { fever: "حمى (Fever)", cough: "سعال (Cough)", shortness_of_breath: "ضيق نفس (Shortness of breath)", sore_throat: "التهاب بالحلق (Sore throat)", runny_nose: "سيلان أنف (Runny nose)", headache: "صداع (Headache)", fatigue: "إعياء (Fatigue)", chest_pain: "ألم صدر (Chest pain)", loss_of_smell_or_taste: "فقدان الشم/الذوق (Loss of smell/taste)", nausea_or_vomiting: "غثيان/قيء (Nausea/Vomiting)", };

// Example rules: conditions are symptoms that must be present for the rule to fire // confidence: arbitrary score (0-1) to help rank multiple diagnoses const RULES = [ { id: "r_covid", name: "احتمال COVID-19", conditions: ["fever", "cough", "loss_of_smell_or_taste"], diagnosis: "Possible COVID-19", confidence: 0.9, explanation: "قواعد مبنية على وجود حمى + سعال + فقدان الشم/الذوق — يطابق نمط COVID-19 الشائع.", }, { id: "r_flu", name: "انفلونزا", conditions: ["fever", "cough", "headache", "fatigue"], diagnosis: "Influenza (Flu)", confidence: 0.8, explanation: "وجود حمى وسعال وصداع وإعياء يميل لأن يكون انفلونزا.", }, { id: "r_common_cold", name: "زكام عادي", conditions: ["runny_nose", "sore_throat", "cough"], diagnosis: "Common Cold", confidence: 0.6, explanation: "أعراض خفيفة مركزة على الأنف والحلق تتناسب مع زكام.", }, { id: "r_pneumonia", name: "التهاب رئوي", conditions: ["fever", "cough", "shortness_of_breath", "chest_pain"], diagnosis: "Pneumonia", confidence: 0.95, explanation: "وجود ضيق نفس + ألم صدر مع سعال وحمى مؤشر قوي لالتهاب رئوي — يتطلب عناية طبية.", }, { id: "r_gastro", name: "مشكلة معوية", conditions: ["nausea_or_vomiting", "fever"], diagnosis: "Gastroenteritis or food-related illness", confidence: 0.5, explanation: "غثيان/قيء مع حمى قد يشير لالتهاب معوي أو تسمم طعامي.", }, ];

// Simple forward-chaining inference engine: returns fired rules and supporting facts function infer(selectedFacts) { const fired = []; for (const rule of RULES) { const matched = rule.conditions.every((c) => selectedFacts.includes(c)); if (matched) { fired.push({ ...rule }); } } // sort by confidence desc fired.sort((a, b) => b.confidence - a.confidence); return fired; }

export default function ExpertDiagnosisApp() { const [selected, setSelected] = useState([]);

const toggle = (sym) => { setSelected((prev) => (prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym])); };

const results = useMemo(() => infer(selected), [selected]);

return ( <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white p-6"> <div className="max-w-5xl mx-auto"> <header className="flex items-center justify-between mb-6"> <h1 className="text-2xl md:text-3xl font-semibold">نظام خبير لتشخيص الأمراض — Expert System</h1> <div className="text-sm text-slate-600">Project: Expert Systems — Team of 10</div> </header>

<main className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Left: symptoms */}
      <section className="col-span-1 md:col-span-2 bg-white p-6 rounded-2xl shadow">
        <h2 className="text-lg font-medium mb-3">اختر الأعراض</h2>
        <p className="text-sm text-slate-500 mb-4">علميًا: كل صندوق يمثل Fact. اختيار مجموعة من الحقائق سيُشغّل الاستدلال.</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {SYMPTOMS.map((s) => (
            <label key={s} className="flex items-center gap-3 p-3 border rounded-lg hover:shadow-sm">
              <input
                type="checkbox"
                checked={selected.includes(s)}
                onChange={() => toggle(s)}
                className="w-4 h-4"
              />
              <div>
                <div className="font-medium">{SYMPTOM_LABEL[s]}</div>
                <div className="text-xs text-slate-500">Fact: {s}</div>
              </div>
            </label>
          ))}
        </div>

        <div className="mt-6">
          <h3 className="font-semibold">Facts الحالية (Selected Facts):</h3>
          <div className="mt-2 flex flex-wrap gap-2">
            {selected.length === 0 ? (
              <span className="text-sm text-slate-500">لا توجد حقائق مختارة</span>
            ) : (
              selected.map((s) => (
                <span key={s} className="px-3 py-1 bg-slate-100 rounded-full text-sm">
                  {SYMPTOM_LABEL[s]}
                </span>
              ))
            )}
          </div>
        </div>
      </section>

      {/* Right: results */}
      <aside className="bg-white p-6 rounded-2xl shadow">
        <h2 className="text-lg font-medium mb-3">النتيجة</h2>

        {results.length === 0 ? (
          <div className="text-sm text-slate-500">لم يتم إيجاد تشخيص مطابق بعد. جرّب إضافة أعراض أكثر.</div>
        ) : (
          <div className="space-y-4">
            {results.map((r) => (
              <div key={r.id} className="border p-3 rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">{r.name}</div>
                    <div className="text-xs text-slate-500">Diagnosis: {r.diagnosis}</div>
                  </div>
                  <div className="text-sm font-medium">Confidence: {(r.confidence * 100).toFixed(0)}%</div>
                </div>
                <div className="mt-2 text-sm text-slate-600">Explanation: {r.explanation}</div>
                <div className="mt-3 text-xs text-slate-500">Used facts:</div>
                <div className="mt-1 flex flex-wrap gap-2">
                  {r.conditions.map((c) => (
                    <span key={c} className="px-2 py-1 bg-slate-100 rounded-full text-xs">
                      {SYMPTOM_LABEL[c]}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="mt-6">
          <h3 className="font-medium">تفاصيل المشروع (قصير)</h3>
          <ul className="text-sm mt-2 list-disc ml-5 text-slate-600">
            <li>Team size: 10 students (mandatory).</li>
            <li>Grading: 5pts idea, 5pts implementation (GUI + engine), 5pts discussion Q&A.</li>
            <li>Inference: forward chaining rule engine (simple, explainable).</li>
          </ul>
        </div>

        <div className="mt-4 flex gap-3">
          <button
            onClick={() => {
              // simple demo: select all symptoms used in top rule to show how it looks
              if (results.length > 0) {
                alert("Rule fired: " + results[0].name + " — see explanation panel.");
              } else {
                alert("لم تطبق أي قاعدة. جرّب اختيار أعراض أخرى.");
              }
            }}
            className="px-4 py-2 rounded-lg bg-slate-800 text-white text-sm"
          >
            Show top rule info
          </button>

          <button
            onClick={() => {
              setSelected([]);
            }}
            className="px-4 py-2 rounded-lg border text-sm"
          >
            Reset
          </button>
        </div>
      </aside>
    </main>

    <section className="mt-8 bg-white p-6 rounded-2xl shadow">
      <h3 className="text-lg font-medium mb-3">قواعد ونِظام العمل المقترح للمشروع</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <h4 className="font-semibold">Roles (أدوار)</h4>
          <ul className="text-sm mt-2 list-disc ml-5 text-slate-600">
            <li>Leader / Coordinator (1): تنسيق الفريق والتواصل معك.</li>
            <li>Domain researcher (2): جمع الحقائق الطبية والمصادر.</li>
            <li>Rules engineer (3): كتابة وصياغة القواعد.</li>
            <li>Frontend dev (2): واجهة المستخدم وUX.</li>
            <li>Backend / integration (2): منطق الاستدلال، حفظ بيانات، وواجهة تجريبية.</li>
          </ul>
        </div>

        <div>
          <h4 className="font-semibold">Deliverables</h4>
          <ol className="text-sm mt-2 list-decimal ml-5 text-slate-600">
            <li>وثيقة فكرة + مخطط القواعد (5 صفحات تقريبا).</li>
            <li>كود التطبيق + واجهة GUI (مع ملفات تشغيل).</li>
            <li>عرض تقديمي + سيناريو مناقشة (أسئلة محتملة).</li>
          </ol>
        </div>

        <div>
          <h4 className="font-semibold">Demo script (نموذج)</h4>
          <ol className="text-sm mt-2 list-decimal ml-5 text-slate-600">
            <li>عرض الفكرة والهدف من النظام (1 دقيقة).</li>
            <li>شرح الحقائق والقواعد (2-3 دقائق).</li>
            <li>تجربة بثلاث حالات مختلفة وشرح لماذا نتحصل على التشخيص (3 دقائق).</li>
            <li>الأسئلة المتوقعة من الدكتور وإجابات قصيرة.</li>
          </ol>
        </div>
      </div>
    </section>

    <footer className="mt-6 text-sm text-slate-500">Need changes? Tell me which parts to adapt: rules, domain (e.g., loans instead of medical), UI style, or export options.</footer>
  </div>
</div>

); }