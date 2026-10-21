import os
import random
from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import db
import notify
from router import (AREAS, DEPARTMENTS, LEVELS, OTHER_LABEL, PROBLEM_TYPES, REPLY_TEMPLATES, SLA_HOURS, TEMPLATES,
                    acknowledgement, apply_resident_choice, build_training,
                    generate_data, train_models, triage)

st.set_page_config(page_title="CityFix | Municipal Service Requests", page_icon="🏙️", layout="wide")
db.init()
for _k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"):
    try:
        os.environ.setdefault(_k, st.secrets[_k])
    except Exception:
        pass

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}
h1, h2, h3 {color: #14253A; letter-spacing: -0.01em;}
.hero {background: linear-gradient(135deg, #14253A 0%, #1F5F6B 100%); color: #F6F3EC; padding: 34px 38px;
       border-radius: 20px; margin-bottom: 22px;}
.hero h1 {color: #F6F3EC; font-size: 2.3rem; margin: 0 0 6px 0;}
.hero p {color: #D5E3E6; font-size: 1.08rem; margin: 0;}
.hero .tag {display: inline-block; background: rgba(233,180,76,.18); color: #E9B44C; padding: 4px 12px;
            border-radius: 999px; font-size: .78rem; font-weight: 700; letter-spacing: .08em; margin-bottom: 12px;}
.card {background: #FFFFFF; color: #1D2B3A; border: 1px solid #E4DED1; border-radius: 16px; padding: 20px 22px;
       box-shadow: 0 2px 10px rgba(20,37,58,.05); height: 100%;}
.card h4 {color: #1F7A6F; margin: 0 0 6px 0; font-size: 1.05rem;}
.card p {margin: 0; color: #3A4A5A; font-size: .95rem; line-height: 1.45;}
.step-num {display: inline-flex; width: 30px; height: 30px; border-radius: 50%; background: #1F7A6F; color: #fff;
           align-items: center; justify-content: center; font-weight: 700; margin-bottom: 10px;}
.receipt {background: #FFFFFF; color: #1D2B3A; border: 2px dashed #1F7A6F; border-radius: 16px; padding: 18px 24px;
          display: flex; gap: 40px; flex-wrap: wrap; margin: 12px 0 6px 0;}
.receipt .lab {font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; color: #6A7179;}
.receipt .big {font-size: 1.9rem; font-weight: 800; color: #14253A; letter-spacing: .04em;}
.badge {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .78rem; font-weight: 700;
        color: #fff; margin-right: 6px;}
.ticket-head {background: #FFFFFF; color: #1D2B3A; border: 1px solid #E4DED1; border-left: 6px solid #1F7A6F;
              border-radius: 14px; padding: 16px 20px; margin: 8px 0 14px 0;}
.ticket-head .ref {font-size: 1.3rem; font-weight: 800; margin-right: 10px;}
.ticket-head .meta {color: #5D646B; font-size: .9rem; margin-top: 6px;}
[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E4DED1; border-radius: 14px; padding: 14px 18px;
                          box-shadow: 0 2px 8px rgba(20,37,58,.04);}
[data-testid="stMetric"] * {color: #1D2B3A !important;}
[data-testid="stMetricValue"] {font-weight: 800;}
.stButton > button, .stFormSubmitButton > button {border-radius: 12px; font-weight: 700; padding: .55rem 1.4rem;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600;}
[data-testid="stChatMessage"] {background: #FFFFFF; color: #1D2B3A; border: 1px solid #E4DED1; border-radius: 16px;
                               padding: 12px 16px; margin-bottom: 10px;}
[data-testid="stChatMessage"] * {color: #1D2B3A;}
[data-testid="stSidebar"] {background: #14253A;}
[data-testid="stSidebar"] * {color: #F6F3EC;}
[data-testid="stSidebar"] input {color: #14253A !important;}
.sms {background: #E7F3EE; color: #1D2B3A; border: 1px solid #BFDCD0; border-radius: 18px 18px 18px 4px;
       padding: 12px 16px; max-width: 520px; margin: 8px 0;}
.sms .lab {font-size: .75rem; color: #4B6B5F; margin-bottom: 4px; font-weight: 700;}
textarea, input {border-radius: 10px !important;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def badge(text: str, colour: str) -> str:
    return f'<span class="badge" style="background:{colour}">{text}</span>'


def hero(tag: str, title: str, sub: str):
    st.markdown(f'<div class="hero"><div class="tag">{tag}</div><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def card(title: str, body: str, num: str = ""):
    n = f'<div class="step-num">{num}</div>' if num else ""
    return f'<div class="card">{n}<h4>{title}</h4><p>{body}</p></div>'


MODEL_PATH = "cityfix_model.joblib"
URG_COLOURS = {"Critical": "#d64545", "High": "#e8913a", "Medium": "#e3c04d", "Low": "#5aa17f"}
STATUSES = ["Open", "Under review", "In progress", "Resolved"]
STATUS_COLOURS = {"Open": "#3B6FB6", "Under review": "#8E5BB5", "In progress": "#E8913A", "Resolved": "#5AA17F"}
CATEGORIES = list(DEPARTMENTS)
CONF_THRESHOLD = 0.55


# ---------------------------------------------------------------- models
@st.cache_resource
def _load(version: float):
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    cat, urg, report = train_models(generate_data())
    return cat, urg, report


def get_models():
    return _load(os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0)


cat_model, urg_model, report = get_models()


def officer_pin() -> str:
    try:
        return st.secrets["OFFICER_PIN"]
    except Exception:
        return "officer123"  # demo default; set OFFICER_PIN in Streamlit secrets for a real deployment


# ---------------------------------------------------------------- demo data
def seed_demo():
    if db.count_tickets() > 0:
        return
    rng = random.Random(5)
    cats = [c for c in TEMPLATES if c != "Unclassified"]
    for i in range(16):
        text = rng.choice(TEMPLATES[rng.choice(cats)]).format(a=rng.choice(AREAS), n=rng.randint(2, 9))
        r = triage(text, cat_model, urg_model, CONF_THRESHOLD)
        when = (datetime.now() - timedelta(hours=rng.randint(2, 90))).strftime("%Y-%m-%d %H:%M")
        status = rng.choice(["Open", "In progress", "Resolved", "Resolved"])
        ref, _ = db.create_ticket(text, rng.choice(AREAS), r, created=when, status=status)
        t = db.get_ticket(ref)
        db.add_message(t["id"], "system", acknowledgement(r, ref), created=when)
        if status in ("In progress", "Resolved"):
            db.add_message(t["id"], "officer", REPLY_TEMPLATES["Team dispatched" if status == "In progress" else "Resolved"], created=when)
            db.update_ticket(t["id"], needs_review=0)


seed_demo()


def render_thread(ticket_id: int):
    for m in db.get_messages(ticket_id):
        if m["sender"] == "resident":
            with st.chat_message("user"):
                st.write(m["body"]); st.caption(f"You · {m['created']}")
        else:
            who, avatar = ("CityFix (automatic)", "🤖") if m["sender"] == "system" else ("Municipal officer", "🧑‍💼")
            with st.chat_message("assistant", avatar=avatar):
                st.write(m["body"]); st.caption(f"{who} · {m['created']}")


# ---------------------------------------------------------------- layout
st.sidebar.markdown("## 🏙️ CityFix")
st.sidebar.caption("Municipal service requests, made simple.")
portal = st.sidebar.radio("Go to", ["🏠 Resident portal", "🧑‍💼 Staff portal"], label_visibility="collapsed")
st.sidebar.caption("Demo mode: please do not enter real personal details.")

# ================================================================ RESIDENT
if portal.endswith("Resident portal"):
    hero("RESIDENT PORTAL", "Report a problem. Get a reply.",
         "Tell us what is wrong, get a reference number straight away, and follow your request until it is fixed.")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Tell us", "Describe the problem and the area in your own words.", "1"), unsafe_allow_html=True)
    c2.markdown(card("Get a reply", "You instantly get a reference number and an expected response time.", "2"), unsafe_allow_html=True)
    c3.markdown(card("Follow up", "Track your request, read officer replies and send more information.", "3"), unsafe_allow_html=True)
    st.write("")
    tab_new, tab_track = st.tabs(["📨 Report a problem", "🔎 Track my request"])

    with tab_new:
        placeholder_type = "Choose a problem type..."
        left, right = st.columns([3, 2])
        with left:
            ptype = st.selectbox("What kind of problem is it?", [placeholder_type] + list(PROBLEM_TYPES) + [OTHER_LABEL])
            other_text = ""
            if ptype == OTHER_LABEL:
                other_text = st.text_input("Please tell us what the problem is", placeholder="e.g. Fallen tree blocking the pavement")
            text = st.text_area("Tell us more, in your own words", height=120,
                                placeholder="What is happening, and since when? The more detail, the faster we can help.")
            street = st.text_input("Street address or nearest landmark",
                                   placeholder="e.g. 12 Vilakazi Street, next to the taxi rank")
            suburb = st.text_input("Suburb or area", placeholder="e.g. Soweto")
            st.markdown("**Get updates by SMS (optional)**")
            phone_raw = st.text_input("Cell phone number", placeholder="e.g. 078 123 4567")
            consent = st.checkbox("You may send me text messages about this request. My number is used only for this.")
            if st.button("Send request", type="primary"):
                phone = notify.normalise_phone(phone_raw) if phone_raw.strip() else None
                if ptype == placeholder_type:
                    st.error("Please choose the kind of problem, or pick Other and tell us what it is.")
                elif ptype == OTHER_LABEL and not other_text.strip():
                    st.error("You chose Other, so please type what the problem is.")
                elif not text.strip():
                    st.error("Please tell us a bit more about the problem.")
                elif not street.strip():
                    st.error("Please tell us where the problem is, with a street address or landmark.")
                elif phone_raw.strip() and phone is None:
                    st.error("That cell phone number does not look right. Use a South African number like 078 123 4567.")
                elif phone and not consent:
                    st.error("Please tick the box to allow text messages, or remove the phone number.")
                else:
                    if ptype == OTHER_LABEL:
                        label, ai_text = f"Other: {other_text.strip()}", f"{other_text.strip()}. {text.strip()}"
                    else:
                        label, ai_text = ptype[2:].strip(), f"{PROBLEM_TYPES[ptype][1]}. {text.strip()}"
                    r = apply_resident_choice(triage(ai_text, cat_model, urg_model, CONF_THRESHOLD), ptype, CONF_THRESHOLD)
                    message = f"{label}: {text.strip()}"
                    address = f"{street.strip()}, {suburb.strip()}" if suburb.strip() else street.strip()
                    ref, secret = db.create_ticket(message, suburb.strip() or "Not given", r, address=address,
                                                   phone=phone, sms_consent=bool(phone and consent), problem_type=label)
                    t = db.get_ticket(ref)
                    db.add_message(t["id"], "resident", message)
                    db.add_message(t["id"], "system", acknowledgement(r, ref))
                    notify.send_sms(t, notify.msg_received(ref, secret, r["urgency"], r["sla_hours"]))
                    st.session_state["last_new"] = (ref, secret)
        with right:
            st.markdown(card("Is someone in danger?", "For fires, gas leaks or anyone hurt, call <b>112</b> from a cell phone first, "
                             "then report it here."), unsafe_allow_html=True)
            st.write("")
            st.markdown(card("Your details", "Your address helps officers find the problem. Your phone number is optional and is only used "
                             "to text you your reference number and updates."), unsafe_allow_html=True)
        if "last_new" in st.session_state:
            ref, secret = st.session_state["last_new"]
            t = db.get_ticket(ref)
            st.success("Your request has been received.")
            st.markdown(f'<div class="receipt"><div><div class="lab">Reference number</div><div class="big">{ref}</div></div>'
                        f'<div><div class="lab">Tracking code</div><div class="big">{secret}</div></div></div>', unsafe_allow_html=True)
            st.caption("Keep both. You need them to check progress and reply.")
            for n in db.get_notifications(t["id"]):
                label = {"sent": "SMS sent", "demo": "SMS preview (demo mode, not actually sent)", "failed": "SMS could not be sent"}[n["status"]]
                st.markdown(f'<div class="sms"><div class="lab">📱 {label} to {notify.mask_phone(t["phone"])}</div>{n["body"]}</div>',
                            unsafe_allow_html=True)
            render_thread(t["id"])

    with tab_track:
        c1, c2 = st.columns(2)
        ref_in = c1.text_input("Reference number", placeholder="CF-1001")
        code_in = c2.text_input("Tracking code", placeholder="4 characters")
        if st.button("Check status", type="primary"):
            st.session_state["track"] = (ref_in, code_in)
        if "track" in st.session_state:
            t = db.get_ticket(*st.session_state["track"])
            if t is None:
                st.error("We could not find a request with that reference number and tracking code.")
            else:
                st.markdown(f'<div class="ticket-head"><span class="ref">{t["ref"]}</span>'
                            f'{badge(t["status"], STATUS_COLOURS[t["status"]])}'
                            f'<div class="meta">{t["category"]} · last update {t["updated"]}</div></div>', unsafe_allow_html=True)
                render_thread(t["id"])
                reply = st.chat_input("Add more information or ask a question")
                if reply:
                    db.add_message(t["id"], "resident", reply)
                    st.rerun()

# ================================================================ STAFF
elif portal.endswith("Staff portal"):
    hero("STAFF PORTAL", "Ticket inbox", "Review AI-sorted requests, reply to residents and correct the AI when it gets something wrong.")
    pin = st.sidebar.text_input("Staff PIN", type="password")
    if pin != officer_pin():
        st.info("Enter the staff PIN in the sidebar to continue. Demo PIN: officer123")
        st.stop()
    tab_in, tab_dash, tab_model = st.tabs(["📥 Inbox", "📊 Dashboard", "🔁 Model & retraining"])
    tickets = db.list_tickets()

    with tab_in:
        f1, f2 = st.columns(2)
        stat = f1.multiselect("Status", STATUSES, default=["Open", "Under review", "In progress"])
        only_reply = f2.checkbox("Only tickets waiting for our reply")
        v = tickets[tickets["status"].isin(stat)]
        if only_reply:
            v = v[v["awaiting_officer"] == 1]
        v = v.assign(_u=v["urgency"].map({u: i for i, u in enumerate(reversed(LEVELS))})).sort_values(
            ["awaiting_officer", "_u", "id"], ascending=[False, True, True])
        show = v[["ref", "created", "urgency", "category", "department", "status", "needs_review", "awaiting_officer"]].copy()
        show["needs_review"] = show["needs_review"].map({1: "⚠️ Check", 0: ""})
        show["awaiting_officer"] = show["awaiting_officer"].map({1: "💬 Reply needed", 0: ""})
        show.columns = ["Ref", "Received", "Urgency", "Category", "Department", "Status", "AI flag", "Resident"]
        st.dataframe(show.reset_index(drop=True), use_container_width=True, hide_index=True)
        if v.empty:
            st.info("No tickets match these filters.")
        else:
            sel = st.selectbox("Open a ticket", v["ref"].tolist())
            t = db.get_ticket(sel)
            st.markdown(f'<div class="ticket-head"><span class="ref">{t["ref"]}</span>'
                        f'{badge(t["urgency"], URG_COLOURS[t["urgency"]])}{badge(t["status"], STATUS_COLOURS[t["status"]])}'
                        f'<div class="meta">{t["category"]} · {("Resident chose: " + t["problem_type"] + " · ") if t.get("problem_type") else ""}{t.get("address") or t["area"]} · AI confidence {t["confidence"]:.0%} · '
                        f'target response {t["sla_hours"]} hrs · received {t["created"]}</div></div>', unsafe_allow_html=True)
            render_thread(t["id"])
            if t.get("phone") and t.get("sms_consent"):
                with st.expander(f"📱 SMS updates to {notify.mask_phone(t['phone'])}"):
                    notes = db.get_notifications(t["id"])
                    for n in notes:
                        st.caption(f"{n['created']} · {n['status']}")
                        st.write(n["body"])
                    if not notes:
                        st.write("No messages sent yet.")
            with st.form(f"reply_{t['ref']}"):
                st.markdown("**Update and reply**")
                a, b, c = st.columns(3)
                new_cat = a.selectbox("Category", CATEGORIES, index=CATEGORIES.index(t["category"]))
                new_urg = b.selectbox("Urgency", LEVELS, index=LEVELS.index(t["urgency"]))
                new_status = c.selectbox("Status", STATUSES, index=STATUSES.index(t["status"]))
                tpl = st.selectbox("Quick reply", list(REPLY_TEMPLATES))
                msg = st.text_area("Reply to resident", placeholder="Write a reply, or pick a quick reply above.")
                if st.form_submit_button("Save and send", type="primary"):
                    if new_cat != t["category"] or new_urg != t["urgency"]:
                        db.log_correction(t["id"], t["text"], t["category"], new_cat, t["urgency"], new_urg)
                    db.update_ticket(t["id"], category=new_cat, urgency=new_urg, status=new_status,
                                     department=DEPARTMENTS[new_cat], sla_hours=SLA_HOURS[new_urg], needs_review=0)
                    body = msg.strip() or REPLY_TEMPLATES[tpl]
                    if body:
                        db.add_message(t["id"], "officer", body)
                    if body or new_status != t["status"]:
                        notify.send_sms(t, notify.msg_update(t["ref"], new_status, body))
                    st.rerun()

    with tab_dash:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Tickets", len(tickets))
        k2.metric("Critical", int((tickets["urgency"] == "Critical").sum()))
        k3.metric("Need human review", int(tickets["needs_review"].sum()))
        k4.metric("Waiting for our reply", int(tickets["awaiting_officer"].sum()))
        k5.metric("Resolved", int((tickets["status"] == "Resolved").sum()))
        a, b = st.columns(2)
        fig1 = px.bar(tickets.groupby(["category", "urgency"]).size().reset_index(name="Tickets"),
                      x="category", y="Tickets", color="urgency", color_discrete_map=URG_COLOURS,
                      title="Tickets by category and urgency", template="plotly_white")
        fig2 = px.pie(tickets, names="department", title="Workload by department", hole=0.5, template="plotly_white",
                      color_discrete_sequence=["#1F7A6F", "#14253A", "#E9B44C", "#5AA17F", "#8E5BB5", "#E8913A"])
        for f in (fig1, fig2):
            f.update_layout(margin=dict(t=50, l=10, r=10, b=10), font=dict(size=13))
        fig1.update_xaxes(title=None)
        a.plotly_chart(fig1, use_container_width=True)
        b.plotly_chart(fig2, use_container_width=True)
        st.download_button("Download tickets (CSV)", tickets.drop(columns=["secret"]).to_csv(index=False), "cityfix_tickets.csv")

    with tab_model:
        corr = db.get_corrections()
        st.markdown(card("Learning from officer corrections",
                         f"Officers have corrected <b>{len(corr)}</b> AI decisions. Retraining adds these corrected tickets to the training data."),
                    unsafe_allow_html=True)
        st.write("")
        if not corr.empty:
            st.dataframe(corr[["created", "text", "old_category", "new_category", "old_urgency", "new_urgency"]],
                         use_container_width=True, hide_index=True)
        if st.button("Retrain models with corrections", type="primary", disabled=corr.empty):
            cat, urg, rep = train_models(build_training(generate_data(), corr))
            joblib.dump((cat, urg, rep), MODEL_PATH)
            st.success("Models retrained and saved. New requests now use the updated models.")
            st.rerun()

