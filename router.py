"""CityFix core logic: synthetic data, classifiers, safety rules, routing."""
import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DEPARTMENTS = {
    "Unclassified": "Manual Triage Desk",
    "Water & Sanitation": "Water & Sanitation Department",
    "Electricity": "City Power / Electricity Department",
    "Roads & Stormwater": "Roads & Transport Department",
    "Waste & Cleansing": "Pikitup / Solid Waste Department",
    "Community Safety": "Public Safety & Emergency Management",
}

AREAS = ["Soweto", "Sandton", "Alexandra", "Randburg", "Roodepoort", "Midrand",
         "Johannesburg CBD", "Orange Farm", "Fourways", "Bryanston"]

# (category, text templates)
TEMPLATES = {
    "Water & Sanitation": [
        "There is a burst water pipe flooding the street in {a}",
        "No water supply in our area for {n} days in {a}",
        "Sewage is overflowing from a manhole near my house in {a}",
        "Leaking water meter wasting water on the pavement in {a}",
        "Blocked drain and sewer smell outside the clinic in {a}",
        "Water is brown and dirty coming out of the taps in {a}",
        "Fire hydrant is leaking continuously in {a}",
    ],
    "Electricity": [
        "Power outage in {a} for {n} days, no electricity",
        "Transformer exploded and sparks are flying in {a}",
        "Fallen power line lying on the road in {a}",
        "Electricity box open with exposed wires near the school in {a}",
        "Prepaid meter not accepting tokens in {a}",
        "Flickering lights and voltage dips every evening in {a}",
        "Illegal electricity connections hanging over the street in {a}",
    ],
    "Roads & Stormwater": [
        "Large pothole damaging cars on the main road in {a}",
        "Traffic lights not working at the intersection in {a}",
        "Sinkhole opened up in the road in {a}",
        "Stormwater drain blocked and the road floods when it rains in {a}",
        "Road sign knocked down and missing in {a}",
        "Faded road markings and broken speed hump in {a}",
        "Bridge railing is broken and unsafe in {a}",
    ],
    "Waste & Cleansing": [
        "Refuse was not collected for {n} weeks in {a}",
        "Illegal dumping of rubble and tyres in the open field in {a}",
        "Overflowing public bins at the taxi rank in {a}",
        "Dead animal on the road needs removal in {a}",
        "Street sweeping has not happened and litter is piling up in {a}",
        "Medical waste dumped near the park in {a}",
        "Garden waste pile left on the pavement in {a}",
    ],
    "Unclassified": [
        "My cat is lost, please help me find it in {a}",
        "I want to know how to apply for a job at the municipality in {a}",
        "Hello, I have a question about my rates account in {a}",
        "Neighbour plays loud music every weekend in {a}",
        "Where can I get a birth certificate in {a}",
        "Thank you for the good service, just testing this form",
        "Can someone call me back about my application",
    ],
    "Community Safety": [
        "Street lights have been out for {n} weeks and it is dangerous at night in {a}",
        "Streetlights not working on our road for {n} weeks in {a}, people are getting mugged",
        "Building on fire with people inside in {a}",
        "Abandoned building used by criminals in {a}",
        "Gas leak smell in the apartment block in {a}",
        "Open manhole with no cover where children play in {a}",
        "Collapsed wall next to the walkway in {a}",
        "Flooded house and people need rescue in {a}",
    ],
}

URGENCY_RULES = {
    "Critical": [
        r"\bfire\b", r"\bexplod", r"\bspark", r"\bfallen power line", r"\bexposed wires?\b",
        r"\bgas leak", r"\bsinkhole", r"\binside\b", r"\brescue", r"\bmedical waste",
        r"\bopen manhole", r"\bcollapse", r"\bsewage\b", r"\bburst\b",
    ],
    "High": [
        r"\bno water", r"street ?lights?", r"\bno electricity", r"\bpower outage", r"\bpothole", r"\btraffic lights",
        r"\bdangerous", r"\bflood", r"\bunsafe", r"\bhydrant", r"\bdirty", r"\bclinic|\bschool|\bhospital",
    ],
    "Medium": [r"\bnot collected", r"\bblocked", r"\billegal", r"\boverflowing", r"\bflicker", r"\bdumping"],
}


def rule_urgency(text: str) -> str:
    """Safety-first keyword rules. These override the ML model upwards, never downwards."""
    t = text.lower()
    for level in ["Critical", "High", "Medium"]:
        if any(re.search(p, t) for p in URGENCY_RULES[level]):
            return level
    return "Low"


def generate_data(n_per_template: int = 12, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    openers = ["", "Please help. ", "Urgent: ", "Good day, ", "Hi, ", "To whom it may concern, "]
    for cat, temps in TEMPLATES.items():
        for tpl in temps:
            for _ in range(n_per_template):
                text = tpl.format(a=rng.choice(AREAS), n=rng.randint(2, 9))
                text = rng.choice(openers) + (text[0].lower() + text[1:] if rng.random() < 0.3 else text)
                rows.append({"text": text, "category": cat, "urgency": rule_urgency(text)})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_models(df: pd.DataFrame):
    X_tr, X_te, yc_tr, yc_te, yu_tr, yu_te = train_test_split(
        df["text"], df["category"], df["urgency"], test_size=0.25, random_state=7, stratify=df["category"])
    make = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                                 LogisticRegression(max_iter=2000, C=5))
    cat_model, urg_model = make().fit(X_tr, yc_tr), make().fit(X_tr, yu_tr)
    report = {
        "category": classification_report(yc_te, cat_model.predict(X_te), output_dict=True, zero_division=0),
        "urgency": classification_report(yu_te, urg_model.predict(X_te), output_dict=True, zero_division=0),
    }
    return cat_model, urg_model, report


LEVELS = ["Low", "Medium", "High", "Critical"]
SLA_HOURS = {"Critical": 2, "High": 24, "Medium": 72, "Low": 168}


def triage(text: str, cat_model, urg_model, conf_threshold: float = 0.55) -> dict:
    cat_proba = cat_model.predict_proba([text])[0]
    cat = cat_model.classes_[cat_proba.argmax()]
    cat_conf = float(cat_proba.max())
    ml_urg = urg_model.predict([text])[0]
    rule_urg = rule_urgency(text)
    # Safety rule: final urgency is the HIGHER of model and rules
    final = max(ml_urg, rule_urg, key=LEVELS.index)
    if cat_conf < 0.40:  # out of scope or too vague: do not guess a department
        cat, final = "Unclassified", rule_urg
    needs_human = cat_conf < conf_threshold or final == "Critical" or cat == "Unclassified"
    reason = []
    if final == "Critical":
        reason.append("Critical safety risk: escalate to a human officer immediately")
    if cat_conf < conf_threshold:
        reason.append(f"Low model confidence ({cat_conf:.0%}): manual check needed")
    top = sorted(zip(cat_model.classes_, cat_proba), key=lambda x: -x[1])[:3]
    return {
        "category": cat, "confidence": cat_conf, "urgency": final, "ml_urgency": ml_urg,
        "rule_urgency": rule_urg, "department": DEPARTMENTS[cat], "sla_hours": SLA_HOURS[final],
        "needs_human_review": needs_human, "review_reason": "; ".join(reason) or "Auto-routed",
        "top3": top,
    }


def acknowledgement(r: dict, ref: str) -> str:
    """Automatic first reply sent to the resident."""
    if r["urgency"] == "Critical":
        return (f"Thank you. Your request ({ref}) has been flagged as URGENT and an officer is looking at it now. "
                "If anyone is in immediate danger, please call 112 from a cell phone.")
    if r["category"] == "Unclassified":
        return (f"Thank you. Your message ({ref}) has been received. A staff member will read it and reply to you "
                f"within {r['sla_hours']} hours.")
    if r["needs_human_review"]:
        return (f"Thank you. Your request ({ref}) has been received. An officer is checking it and will reply "
                f"within {r['sla_hours']} hours.")
    return (f"Thank you. Your request ({ref}) has been logged as {r['category']} and sent to the {r['department']}. "
            f"Priority: {r['urgency']}. You can expect a first response within {r['sla_hours']} hours. "
            "You can check progress or add more information at any time with your reference number and tracking code.")


REPLY_TEMPLATES = {
    "(none)": "",
    "Team dispatched": "Thank you for reporting this. A team has been dispatched to the location and we will update you once the work is done.",
    "Need more info": "Thank you. Could you please send the exact street address or the nearest landmark, and tell us when the problem started?",
    "Resolved": "The problem you reported has been fixed. If it happens again, please reply here and we will reopen your request.",
    "Redirected": "Your request has been passed to the correct department. They will contact you with an update.",
}


def build_training(base: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Add officer-corrected tickets to the training data (repeated so each correction carries weight)."""
    if corrections is None or corrections.empty:
        return base
    extra = pd.DataFrame({"text": corrections["text"], "category": corrections["new_category"],
                          "urgency": corrections["new_urgency"]})
    return pd.concat([base, extra, extra, extra], ignore_index=True)


# Problem types residents can pick: label -> (category, words fed to the classifier with the description)
PROBLEM_TYPES = {
    "💧 Burst pipe or water leak": ("Water & Sanitation", "burst water pipe leaking water"),
    "💧 No water supply": ("Water & Sanitation", "no water supply in our area"),
    "💧 Sewage overflow or blocked sewer": ("Water & Sanitation", "sewage overflowing from manhole blocked sewer"),
    "💧 Dirty or discoloured water": ("Water & Sanitation", "water is brown and dirty from the taps"),
    "⚡ Power outage": ("Electricity", "power outage no electricity"),
    "⚡ Fallen or damaged power line": ("Electricity", "fallen power line lying on the road"),
    "⚡ Open or faulty electrical box": ("Electricity", "electricity box open with exposed wires"),
    "⚡ Prepaid meter problem": ("Electricity", "prepaid meter not accepting tokens"),
    "🚧 Pothole": ("Roads & Stormwater", "large pothole damaging cars on the road"),
    "🚧 Traffic lights not working": ("Roads & Stormwater", "traffic lights not working at the intersection"),
    "🚧 Blocked drain or flooded road": ("Roads & Stormwater", "stormwater drain blocked and the road floods"),
    "🚧 Damaged road sign or marking": ("Roads & Stormwater", "road sign knocked down and missing"),
    "🚧 Sinkhole": ("Roads & Stormwater", "sinkhole opened up in the road"),
    "🗑️ Refuse not collected": ("Waste & Cleansing", "refuse was not collected for weeks"),
    "🗑️ Illegal dumping": ("Waste & Cleansing", "illegal dumping of rubble in the open field"),
    "🗑️ Overflowing public bin": ("Waste & Cleansing", "overflowing public bins"),
    "🗑️ Dead animal on the road": ("Waste & Cleansing", "dead animal on the road needs removal"),
    "🛡️ Streetlight not working": ("Community Safety", "street lights not working and it is dangerous at night"),
    "🛡️ Open manhole or unsafe structure": ("Community Safety", "open manhole with no cover where children play"),
    "🛡️ Gas leak": ("Community Safety", "gas leak smell in the apartment block"),
    "🛡️ Fire or emergency": ("Community Safety", "building on fire with people inside"),
}
OTHER_LABEL = "❓ Other (please specify)"


def apply_resident_choice(r: dict, problem_label: str, conf_threshold: float = 0.55) -> dict:
    """Combine the resident's chosen problem type with the AI result.
    - If the AI is unsure or says Unclassified, trust the resident's category but ask a human to check.
    - If the AI confidently disagrees with the resident, keep the AI answer but ask a human to check."""
    choice = PROBLEM_TYPES.get(problem_label)
    if not choice:
        return r
    expected = choice[0]
    if r["category"] == expected:
        return r
    r = dict(r)
    if r["category"] == "Unclassified" or r["confidence"] < conf_threshold:
        r["category"], r["department"] = expected, DEPARTMENTS[expected]
        note = f"Resident chose '{problem_label[2:].strip()}' and the AI was unsure"
    else:
        note = f"AI says {r['category']} but the resident chose '{problem_label[2:].strip()}'"
    r["needs_human_review"] = True
    r["review_reason"] = (r["review_reason"] + "; " if r["review_reason"] != "Auto-routed" else "") + note
    return r
