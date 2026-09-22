"""
Clinical verification dataset for the AfyaPlus evaluation pipeline.

15 questions, balanced across:
  - 3 clinical features: triage_routing, insurance_verification, medication_calculation
  - 3 channels: USSD, Mobile App, Web Portal   (5 each)

Each question is paired with a clinical_reference answer grounded in the
real AfyaPlus knowledge base (see week-2/afyaplus-rag-agent/knowledge/*.md),
so automated metrics (BLEU/ROUGE/F1) and the LLM judge have something
factually correct to compare model output against.
"""

EVALUATION_DATASET = [
    # ---------------- triage_routing (5) ----------------
    {
        "id": "TR-01",
        "feature": "triage_routing",
        "channel": "USSD",
        "question": "I am having severe chest pain and I can't catch my breath.",
        "clinical_reference": (
            "This should be routed to emergency care immediately. Severe chest "
            "pain and severe difficulty breathing are both listed emergency "
            "symptoms that require urgent action."
        ),
    },
    {
        "id": "TR-02",
        "feature": "triage_routing",
        "channel": "Mobile App",
        "question": "My father suddenly can't move the left side of his face and is slurring his words.",
        "clinical_reference": (
            "This should be routed to emergency care immediately. Facial "
            "weakness and difficulty speaking are signs of stroke, which "
            "require immediate emergency routing."
        ),
    },
    {
        "id": "TR-03",
        "feature": "triage_routing",
        "channel": "Web Portal",
        "question": "I have had a persistent high fever for three days and now I'm vomiting repeatedly and feel dehydrated.",
        "clinical_reference": (
            "This patient should receive urgent clinical review. Persistent "
            "high fever combined with persistent vomiting and signs of "
            "dehydration are urgent clinical review indicators, not routine care."
        ),
    },
    {
        "id": "TR-04",
        "feature": "triage_routing",
        "channel": "USSD",
        "question": "I just have a mild cold with a runny nose, nothing else.",
        "clinical_reference": (
            "This can be routed to routine outpatient care. Mild cold "
            "symptoms with no worsening or emergency indicators are a "
            "routine care example."
        ),
    },
    {
        "id": "TR-05",
        "feature": "triage_routing",
        "channel": "Mobile App",
        "question": "My child had a seizure that has now lasted about seven minutes.",
        "clinical_reference": (
            "This should be routed to emergency care immediately. Seizures "
            "lasting more than five minutes are an explicit emergency symptom."
        ),
    },
    # ---------------- insurance_verification (5) ----------------
    {
        "id": "IV-01",
        "feature": "insurance_verification",
        "channel": "Web Portal",
        "question": "Is an outpatient MRI covered by my AfyaPlus plan, and do I need approval first?",
        "clinical_reference": (
            "Outpatient MRI scans are covered when requested by a registered "
            "physician, but they require pre-authorisation before the "
            "procedure. The standard outpatient MRI co-payment is KES 2,000."
        ),
    },
    {
        "id": "IV-02",
        "feature": "insurance_verification",
        "channel": "USSD",
        "question": "I need an emergency CT scan right now, do I need pre-authorisation?",
        "clinical_reference": (
            "No. Emergency CT scans may proceed without prior authorisation "
            "when the attending clinician determines that immediate imaging "
            "is medically necessary."
        ),
    },
    {
        "id": "IV-03",
        "feature": "insurance_verification",
        "channel": "Mobile App",
        "question": "Can I see a specialist without a referral from my regular doctor?",
        "clinical_reference": (
            "Routine specialist consultations require a valid referral from "
            "a registered primary care clinician. Emergency specialist "
            "consultations do not require a referral."
        ),
    },
    {
        "id": "IV-04",
        "feature": "insurance_verification",
        "channel": "Web Portal",
        "question": "Do I need pre-authorisation for a routine blood test ordered by my doctor?",
        "clinical_reference": (
            "Routine laboratory tests are covered when ordered by a "
            "registered clinician and generally do not need pre-authorisation. "
            "Specialised laboratory tests may require pre-authorisation "
            "depending on the patient's benefit plan."
        ),
    },
    {
        "id": "IV-05",
        "feature": "insurance_verification",
        "channel": "USSD",
        "question": "What is my co-payment for an outpatient MRI scan?",
        "clinical_reference": (
            "The standard outpatient MRI co-payment under the AfyaPlus "
            "policy is KES 2,000."
        ),
    },
    # ---------------- medication_calculation (5) ----------------
    {
        "id": "MC-01",
        "feature": "medication_calculation",
        "channel": "Mobile App",
        "question": "The doctor prescribed 500 mg of a medication, and the available concentration is 250 mg/mL. What volume should be administered?",
        "clinical_reference": (
            "Using volume_ml = prescribed_dose_mg / concentration_mg_per_ml: "
            "500 / 250 = 2.00 mL should be administered."
        ),
    },
    {
        "id": "MC-02",
        "feature": "medication_calculation",
        "channel": "Web Portal",
        "question": "A patient needs 750 mg of a drug, and the concentration on hand is 250 mg/mL. How many mL should I give?",
        "clinical_reference": (
            "Using volume_ml = prescribed_dose_mg / concentration_mg_per_ml: "
            "750 / 250 = 3.00 mL should be administered."
        ),
    },
    {
        "id": "MC-03",
        "feature": "medication_calculation",
        "channel": "USSD",
        "question": "Prescribed dose is 100 mg, but the concentration is not specified. Can you calculate the volume?",
        "clinical_reference": (
            "No. The calculation cannot proceed because the concentration is "
            "missing. Missing clinical values must be clarified instead of "
            "guessed; the volume should not be estimated."
        ),
    },
    {
        "id": "MC-04",
        "feature": "medication_calculation",
        "channel": "Mobile App",
        "question": "The dose ordered is 0 mg. What volume should I administer?",
        "clinical_reference": (
            "The calculation cannot be performed for a zero dose. Zero or "
            "negative doses are invalid and clarification should be "
            "requested rather than returning a computed volume."
        ),
    },
    {
        "id": "MC-05",
        "feature": "medication_calculation",
        "channel": "Web Portal",
        "question": "For a dose of 1000 mg with a concentration of 200 mg/mL, what volume is needed, and what unit should be stated?",
        "clinical_reference": (
            "Using volume_ml = prescribed_dose_mg / concentration_mg_per_ml: "
            "1000 / 200 = 5.00 mL. The unit (mL) should always be clearly "
            "stated in the answer."
        ),
    },
]

FEATURES = ["triage_routing", "insurance_verification", "medication_calculation"]
CHANNELS = ["USSD", "Mobile App", "Web Portal"]

assert len(EVALUATION_DATASET) == 15
assert all(q["feature"] in FEATURES for q in EVALUATION_DATASET)
assert all(q["channel"] in CHANNELS for q in EVALUATION_DATASET)
