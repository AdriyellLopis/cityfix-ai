# CityFix: Municipal Service Request Router

**Week 4 AI Solution | CAPACITI Bootcamp | Adriyell Lopis**

An end-to-end AI workflow that reads resident complaints (water leaks, potholes, power outages, refuse, safety hazards), classifies the **type** and **urgency**, routes each to the correct municipal department with a response-time target, and escalates risky or uncertain cases to a human officer.

## The problem
Municipal call centres receive thousands of free-text complaints. Manual sorting is slow, inconsistent and delays dangerous issues (fallen power lines, burst mains) while minor requests clog queues.

## The solution: two-way communication
| Step | What happens |
|---|---|
| 1. Resident reports | Resident types the problem in their own words, adds a street address or landmark and suburb, and can add a cell phone number (with consent) in the **Resident portal** |
| 2. AI triage | TF-IDF + Logistic Regression predict category and urgency; safety rules can only raise urgency |
| 3. Instant response | The resident immediately receives a reference number, a tracking code, an SMS (if a number was given) and an automatic reply (department, priority, expected response time). Urgent cases get a safety message |
| 4. Routing | Confident, non-critical tickets go to a department with a response target. Critical, uncertain or off-topic tickets go to a human officer |
| 5. Officer reply | In the **Staff portal** (PIN protected) officers open the ticket, correct the AI if needed, change the status, and write a reply or use a quick reply |
| 6. Resident follows up | The resident tracks the ticket with their reference and tracking code, reads the officer's reply, and can reply back. A reply flags the ticket for staff and reopens it if it was resolved |
| 7. Learning | Officer corrections are logged and can be used to retrain the models from the Staff portal |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `router.py`: data generation, models, safety rules, triage logic
- `app.py`: Streamlit app (resident portal, staff portal, about and responsible AI)
- `db.py`: SQLite ticket store (tickets, message threads, officer corrections, SMS log)
- `notify.py`: phone number checks and SMS sending (demo mode or live via Twilio)

## Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repo (e.g. `cityfix-ai`).
2. Go to share.streamlit.io, click **New app**, pick the repo, set main file to `app.py`, deploy.

## Using the demo
- Resident portal: report a problem, note the reference number and tracking code, then use **Track my request**.
- Staff portal: enter the PIN (demo default `officer123`; set `OFFICER_PIN` in Streamlit secrets for a real deployment), open a ticket and reply.
- The database starts with 16 simulated tickets so the dashboard is not empty.

## SMS updates
- **Demo mode (default):** nothing is sent. The app shows a preview of the text message and keeps a log, so you can demonstrate the flow for free.
- **Live mode:** add `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `TWILIO_FROM` to the app's Streamlit secrets. Texts are then sent when a ticket is created and when an officer replies or changes the status. A South African provider (Clickatell, BulkSMS, SMSPortal) can be used by editing `_send_live()` in `notify.py`. SMS costs money, and a trial Twilio account can only text numbers you have verified.
- Texts are only sent if the resident gives consent. Staff see the phone number masked.
- Phone numbers and addresses are personal information under POPIA: collect only what is needed, get consent, and set a retention period.

## Honest limitations
- Data is stored in a local SQLite file. On free Streamlit hosting it resets when the app restarts. A real deployment needs a hosted database.
- SMS runs in demo mode until an SMS provider is configured. There is no WhatsApp or email notification yet.
- Training data is **synthetic and templated**, so the ~100% accuracy shown is not realistic. Real accuracy must be tested on real labelled tickets.
- English only; no location, image or duplicate detection yet.

## Next steps
Real ticket data (anonymised), multilingual support (isiZulu, Sesotho, Afrikaans), geolocation and duplicate clustering (unsupervised learning), WhatsApp or email alerts, hosted database.
