"""Generate synthetic Sri Lankan job / study / migration offers (scam and legit) in EN, Sinhala and Singlish.

Writes:
  data/synthetic/synthetic_train.csv   in-distribution offers (train pool)
  data/synthetic/synthetic_test.csv    held-out offers: about half the phrases come from a reserved pool never used in train
  data/synthetic/shifted.csv           drift set: Tamil text, new tactics (crypto, digital-ID fees, cyber-scam compounds,
                                       QR payments, TikTok/Telegram) and new destinations, mixed with ordinary offers

Columns: text, label (scam|legit), category, language, source, tactic
Everything is fictional: agency names, licence numbers and phone numbers (07X XXX XXXX) are placeholders.
Deterministic: same seed -> same files.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synthetic"

CATEGORIES = ["local_job", "foreign_job", "student_visa", "visa_pr"]

# ---------------------------------------------------------------- slot values
SLOTS = {
    "foreign_job": {
        "country": ["Qatar", "Kuwait", "Dubai", "Saudi Arabia", "Oman", "Israel", "Japan", "South Korea",
                    "Romania", "Poland", "Malta", "Cyprus", "Croatia", "Maldives"],
        "job": ["housemaid", "heavy vehicle driver", "construction helper", "factory worker", "hotel cleaner",
                "caregiver", "welder", "staff nurse", "farm worker", "warehouse packer", "kitchen helper", "electrician"],
    },
    "student_visa": {
        "country": ["Australia", "the UK", "Canada", "Japan", "New Zealand", "Germany", "Latvia", "Lithuania",
                    "Malaysia", "Ireland"],
        "job": ["Diploma in Hospitality", "Bachelor of IT", "Nursing degree", "MBA", "Foundation programme",
                "Japanese language course", "Master of Data Science", "Diploma in Business"],
    },
    "visa_pr": {
        "country": ["Canada", "Australia", "New Zealand", "Portugal", "the UK", "Germany", "Ireland"],
        "job": ["Express Entry PR", "skilled migration PR", "family sponsorship", "work permit", "golden visa",
                "job seeker visa", "Blue Card"],
    },
    "local_job": {
        "country": ["Colombo", "Kandy", "Gampaha", "Kurunegala", "Galle", "Negombo", "Matara", "online"],
        "job": ["data entry operator", "packing assistant", "online typist", "sales executive", "cashier",
                "accounts assistant", "delivery rider", "call centre agent", "receptionist", "store keeper"],
    },
}
SCAM_SALARY = {
    "foreign_job": ["Rs. 450,000", "Rs. 600,000", "USD 3,000", "Rs. 850,000", "QAR 9,000", "USD 4,500",
                    "Rs 600k", "6 lakhs", "රු. 450,000"],
    "local_job": ["Rs. 5,000 per day", "Rs. 150,000 per month", "Rs. 8,000 daily", "Rs. 250,000 monthly",
                  "Rs 5k per day", "1.5 lakhs per month", "රු. 8,000 daily"],
    "student_visa": ["full scholarship", "100% scholarship", "free tuition", "Rs. 400,000 part-time income"],
    "visa_pr": ["USD 5,000 monthly", "CAD 6,000 per month", "AUD 7,500 monthly"],
}
LEGIT_SALARY = {
    "foreign_job": ["QAR 1,400", "KWD 120", "AED 1,500", "SAR 1,500", "USD 1,100", "JPY 180,000", "EUR 900",
                    "Rs. 1.2 lakhs", "Rs 120k", "රු. 120,000"],
    "local_job": ["Rs. 55,000 + EPF/ETF", "Rs. 70,000 + EPF/ETF", "Rs. 48,000 + allowances", "Rs. 85,000 negotiable",
                  "Rs 65k + EPF/ETF", "රු. 60,000 + EPF/ETF", "0.75 lakhs + EPF"],
    "student_visa": ["partial merit scholarship", "25% tuition waiver for eligible students", "standard tuition fees"],
    "visa_pr": ["as per official points table", "subject to official assessment", "as per immigration rules"],
}
SCAM_FEE = ["Rs. 150,000", "Rs. 75,000", "Rs. 250,000", "Rs. 45,000", "Rs. 350,000", "Rs. 25,000", "USD 800",
            "Rs 150k", "1.5 lakhs", "රු. 75,000", "LKR 95,000", "2 lakhs"]
LEGIT_FEE = ["the official SLBFE registration fee", "fees as published by the university",
             "the government visa fee only", "the official application fee"]
AGENCY = ["Lanka Bright Overseas (Pvt) Ltd", "Ceylon Horizon Manpower", "Blue Lotus Foreign Employment",
          "Serendib Global Careers", "Island Star Recruiters", "Kelani Gateway Agency", "Ruhunu Pathway Consultants",
          "Pearl Coast Education", "Sigiri Migration Services", "Dewdrop Study Abroad"]
LIC = ["1023", "1187", "1542", "2051", "2316", "0987", "1764", "2290"]
# Each agency's own licence number (the first six match data/kb/registry_demo.csv; Island Star is suspended there).
# Legit ads quote their own agency's licence; scams may borrow a real-looking number.
AGENCY_LICENCE = {"Lanka Bright Overseas (Pvt) Ltd": "1023", "Ceylon Horizon Manpower": "1187",
                  "Blue Lotus Foreign Employment": "1542", "Serendib Global Careers": "2051",
                  "Kelani Gateway Agency": "2316", "Island Star Recruiters": "0987",
                  "Ruhunu Pathway Consultants": "3104", "Pearl Coast Education": "3220",
                  "Sigiri Migration Services": "3377", "Dewdrop Study Abroad": "3412"}
LEGIT_AGENCIES = [a for a in AGENCY if a != "Island Star Recruiters"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# ---------------------------------------------------------------- phrase pools
# Pool layout: POOLS[lang][kind][key] -> list of templates. kind in open|offer|scam|legit|neutral|close.
# For pools with >= 3 phrases, every 3rd phrase (index % 3 == 2) is reserved for the held-out test split.
POOLS = {
    "en": {
        "open": {
            "scam": ["URGENT!!!", "Golden opportunity!", "Last chance - apply now!", "BIG NEWS for job seekers!",
                     "Don't miss this!", "Life changing offer", "Only for the first 20 applicants!", "HURRY UP"],
            "legit": ["Vacancies", "Job announcement", "We are recruiting", "Notice to applicants",
                      "Applications invited", "Career opportunity", "Recruitment notice", "Now hiring"],
        },
        "offer": {
            "foreign_job": ["{job} jobs in {country}, salary {salary} per month.",
                            "Immediate vacancies for {job}s in {country}. Monthly salary {salary}.",
                            "{country} needs 50 {job}s. Salary up to {salary}.",
                            "Work in {country} as a {job}, earn {salary}.",
                            "Overseas opening: {job} - {country} - {salary}."],
            "student_visa": ["Study {job} in {country} with {salary}.",
                             "Intake open for {job} in {country}, {salary} available.",
                             "Get admission to a {job} in {country}. {salary}.",
                             "{country} {job} - apply for the next intake, {salary}.",
                             "Student visa for {country}: {job}, {salary}."],
            "visa_pr": ["Migrate to {country} through {job}. Expected income {salary}.",
                        "{country} {job} - {salary}.",
                        "Apply for {country} {job} with our help. {salary}.",
                        "Your family can move to {country} via {job}, {salary}.",
                        "New {country} {job} pathway, {salary}."],
            "local_job": ["{job} needed in {country}. {salary}.",
                          "Hiring {job}s - {country} - {salary}.",
                          "Vacancy: {job} ({country}), salary {salary}.",
                          "Earn {salary} as a {job}, {country}.",
                          "{job} position available in {country}, {salary}."],
        },
        "scam": {
            "any": ["Pay the registration fee of {fee} today to confirm your seat.",
                    "100% guaranteed visa, no rejection.",
                    "No interview needed, no experience required.",
                    "Deposit {fee} to the personal account below before Friday.",
                    "Only 5 slots left, offer closes today.",
                    "Send your passport copy and NIC photo on WhatsApp to reserve.",
                    "Processing fee {fee} must be paid first, refundable later.",
                    "Send the advance payment via eZ Cash to this number.",
                    "Do not tell other agents about this offer.",
                    "Visa approval guaranteed within 7 days.",
                    "Pay now, fly next week.",
                    "Our manager will collect cash at the bus stand."],
            "foreign_job": ["Go on a visit visa and we will convert it to a work visa there.",
                            "No need to register with SLBFE, we handle everything.",
                            "Free air ticket after paying the deposit of {fee}.",
                            "Salary {salary} plus free food, free accommodation, overtime unlimited."],
            "student_visa": ["No IELTS required, guaranteed admission.",
                             "Bank statement arranged by us, just pay {fee}.",
                             "Work full time while studying, earn {salary}.",
                             "Visa 100% guaranteed even with low results."],
            "visa_pr": ["Direct PR without points test.",
                        "Guaranteed PR in 3 months, skip the points system.",
                        "We have contacts inside the embassy, pay {fee} for fast approval.",
                        "Job offer letter plus PR package for {fee}."],
            "local_job": ["Like YouTube videos and earn daily commission.",
                          "Complete prepaid tasks, pay {fee} to unlock VIP level.",
                          "Rate hotels online and earn {salary}.",
                          "Join our Telegram group, first task bonus Rs. 2,000.",
                          "Work from home 1 hour a day, no qualifications needed."],
        },
        "legit": {
            "any": ["Interviews will be held at our office on {day}.",
                    "No payment is required at the application stage.",
                    "Selected candidates will sign the contract at the office.",
                    "Applicants must apply through the official portal.",
                    "Please bring original certificates for verification.",
                    "Shortlisted candidates will be contacted by email.",
                    "Beware of fraudsters asking for money in our name.",
                    "Do not pay anyone to get this job."],
            "foreign_job": ["SLBFE licence no. {lic}. Registration with SLBFE is compulsory.",
                            "Only {legit_fee} is payable, receipts issued.",
                            "Job order approved by SLBFE, contract attested by the embassy.",
                            "Pre-departure training at an SLBFE centre is required.",
                            "Medical test at an approved centre before departure."],
            "student_visa": ["Minimum IELTS 6.0 required for admission.",
                             "Offer letter issued directly by the university.",
                             "Visa decision is made only by the embassy, we cannot guarantee it.",
                             "Tuition is paid directly to the university, not to us.",
                             "Free counselling session, {legit_fee} only."],
            "visa_pr": ["Eligibility is assessed under the official points system.",
                        "We are a registered migration adviser, licence {lic}.",
                        "Visa outcomes depend on the immigration department.",
                        "Check the official immigration website for current rules.",
                        "Processing times vary, no guarantees are given."],
            "local_job": ["EPF and ETF provided.",
                          "Send your CV to the HR department.",
                          "Minimum qualification: 6 passes in GCE O/L.",
                          "Training will be provided at the {country} branch.",
                          "Walk-in interview from 9 am to 3 pm."],
        },
        "contact": ["Call or WhatsApp 07X XXX XXXX.", "{agency}, Tel: 011 XXX XXXX.", "Email: jobs@example.lk",
                    "Visit our office in {country}.", "Hotline 07X XXX XXXX.", "Inbox us for details.",
                    "{agency} - call 07X XXX XXXX.", "More info: 011 XXX XXXX / 07X XXX XXXX."],
        "neutral": ["Contact us on WhatsApp 07X XXX XXXX.", "Limited vacancies.", "Call 07X XXX XXXX for details.",
                    "Apply early.", "Male and female candidates can apply.", "Age 21 - 45."],
        "close": {
            "scam": ["Call or WhatsApp now 07X XXX XXXX!!!", "Inbox us for details.",
                     "Contact agent Ruwan on Viber 07X XXX XXXX.", "DM now before it is too late!",
                     "Telegram: @fastjobslk", "Hurry, send your details now."],
            "legit": ["{agency}, {country} branch. Tel: 011 XXX XXXX.", "Apply via email: careers@example.lk",
                      "{agency} - registered with SLBFE (licence {lic}).", "Office hours 9 am - 5 pm, weekdays.",
                      "For more details visit our office.", "Apply before the closing date on our website."],
        },
    },
    "singlish": {
        "open": {
            "scam": ["Hadisi!!!", "Supiri chance ekak!", "Ada witharai!", "Miss karanna epa!", "Aththa ekak!!"],
            "legit": ["Rassa awastha", "Purappadu", "Ayadum kandavaganna", "Awasthawak"],
        },
        "offer": {
            "foreign_job": ["{country} wala {job} rassawal, padi {salary}.",
                            "{country} yanna {job} kenek ona, masika padi {salary}.",
                            "{job} job {country}, {salary} gewanawa.",
                            "{country} rassawak {job} widihata, {salary}."],
            "student_visa": ["{country} study karanna {job}, {salary}.",
                             "{country} {job} intake eka open, {salary}.",
                             "Student visa {country} - {job} - {salary}.",
                             "{job} karanna {country} yanna, {salary}."],
            "visa_pr": ["{country} PR ganna {job} through, {salary}.",
                        "{country} {job} pavula ekkama yanna, {salary}.",
                        "{job} {country} lesiyen, {salary}.",
                        "{country} settle wenna {job}, {salary}."],
            "local_job": ["{country} {job} kenek ona, {salary}.",
                          "{job} rassawa {country}, padi {salary}.",
                          "Gedara inna gaman {job}, {salary}.",
                          "{job} awashyai {country}, {salary}."],
        },
        "scam": {
            "any": ["Registration fee eka {fee} ada ma gewanna.",
                    "Visa eka 100% guarantee.",
                    "Interview ekak na, experience ona na.",
                    "Salli personal account ekata dapan.",
                    "Seats 5i thiyenne, ikmanata.",
                    "Passport copy eka NIC eka WhatsApp karanna.",
                    "eZ Cash walin advance eka yawanna.",
                    "Kisi kenekuta kiyanna epa."],
            "foreign_job": ["Visit visa eken gihin work visa ekata convert karanawa.",
                            "SLBFE register wenna ona na.",
                            "Deposit eka {fee} gewwama ticket eka free."],
            "student_visa": ["IELTS ona na, admission guarantee.",
                             "Bank statement api hadala denawa, {fee} witharai.",
                             "Study karana gaman full time wada."],
            "visa_pr": ["Points ona na, direct PR.",
                        "Masa 3n PR guarantee.",
                        "Embassy eke api dannawa, {fee} dunnama ikmanata."],
            "local_job": ["YouTube videos like karala commission ganna.",
                          "Task ekak karanna {fee} gewala VIP wenna.",
                          "Gedara idan dawasata {salary} hambu karanna.",
                          "Telegram group ekata join wenna, bonus Rs. 2,000."],
        },
        "legit": {
            "any": ["Interview eka {day} api office eke.",
                    "Apply karaddi salli gewanna ona na.",
                    "Original certificates aran enna.",
                    "Kisima kenekuta salli denna epa."],
            "foreign_job": ["SLBFE licence {lic}. SLBFE register wenna ona.",
                            "{legit_fee} witharai, receipt denawa.",
                            "Job order eka SLBFE approve karala."],
            "student_visa": ["IELTS 6.0 ona.", "Offer letter eka university eken ma.",
                             "Visa eka embassy eken thamai decide karanne."],
            "visa_pr": ["Official points system eken eligibility check karanawa.",
                        "Registered migration adviser, licence {lic}.",
                        "Guarantee ekak denna ba, immigration eka thamai decide karanne."],
            "local_job": ["EPF ETF thiyenawa.", "CV eka HR ekata email karanna.",
                          "O/L pass wela thiyenna ona."],
        },
        "contact": ["Call / WhatsApp 07X XXX XXXX.", "{agency}, Tel 011 XXX XXXX.", "Inbox karanna.",
                    "Office ekata enna, {country}.", "Email: jobs@example.lk"],
        "neutral": ["WhatsApp 07X XXX XXXX.", "Call karanna 07X XXX XXXX.", "Purappadu tikai.",
                    "Pirimi gahanu dennatama puluwan."],
        "close": {
            "scam": ["Dan ma call karanna 07X XXX XXXX!!!", "Inbox karanna.", "Viber 07X XXX XXXX."],
            "legit": ["{agency}, Tel 011 XXX XXXX.", "Email: careers@example.lk",
                      "{agency} - SLBFE licence {lic}."],
        },
    },
    "si": {
        "open": {
            "scam": ["හදිසියි!!!", "රන් අවස්ථාවක්!", "අද පමණයි!", "මග හරින්න එපා!"],
            "legit": ["පුරප්පාඩු", "රැකියා අවස්ථා", "අයදුම්පත් කැඳවීම", "බඳවා ගැනීම්"],
        },
        "offer": {
            "foreign_job": ["{country} සඳහා {job} පුරප්පාඩු, මාසික වැටුප {salary}.",
                            "{country} හි {job} රැකියා, වැටුප {salary}.",
                            "{job} ලෙස {country} රැකියාවක්, {salary}.",
                            "විදේශ රැකියා: {country} - {job} - {salary}."],
            "student_visa": ["{country} හි {job} හැදෑරීමට අවස්ථාව, {salary}.",
                             "{country} ශිෂ්‍ය වීසා - {job}, {salary}.",
                             "{job} සඳහා {country} විශ්වවිද්‍යාල ප්‍රවේශය, {salary}.",
                             "{country} උසස් අධ්‍යාපනය: {job}, {salary}."],
            "visa_pr": ["{job} හරහා {country} ස්ථිර පදිංචිය, {salary}.",
                        "{country} {job} සඳහා අයදුම් කරන්න, {salary}.",
                        "පවුල සමඟ {country} පදිංචියට {job}, {salary}.",
                        "{country} PR - {job} - {salary}."],
            "local_job": ["{country} {job} අවශ්‍යයි, {salary}.",
                          "{job} පුරප්පාඩුව - {country} - {salary}.",
                          "ගෙදර සිට {job} රැකියාව, {salary}.",
                          "{job} සඳහා බඳවා ගැනේ ({country}), {salary}."],
        },
        "scam": {
            "any": ["ලියාපදිංචි ගාස්තුව {fee} අදම ගෙවන්න.",
                    "100% වීසා සහතිකයි.",
                    "සම්මුඛ පරීක්ෂණයක් අවශ්‍ය නැත, පළපුරුද්ද අවශ්‍ය නැත.",
                    "මුදල් පෞද්ගලික ගිණුමට බැර කරන්න.",
                    "ස්ථාන 5ක් පමණයි, අදම ලියාපදිංචි වන්න.",
                    "විදේශ ගමන් බලපත්‍රයේ සහ හැඳුනුම්පතේ පිටපත් WhatsApp කරන්න.",
                    "eZ Cash හරහා අත්තිකාරම් මුදල එවන්න.",
                    "මෙම දීමනාව වෙනත් අයට කියන්න එපා."],
            "foreign_job": ["සංචාරක වීසා මත ගොස් රැකියා වීසා බවට පරිවර්තනය කරනු ලැබේ.",
                            "SLBFE ලියාපදිංචිය අවශ්‍ය නැත.",
                            "{fee} තැන්පතුව ගෙවූ පසු ගුවන් ටිකට්පත නොමිලේ."],
            "student_visa": ["IELTS අවශ්‍ය නැත, ප්‍රවේශය සහතිකයි.",
                             "බැංකු ප්‍රකාශය අපි සකස් කර දෙන්නෙමු, {fee} පමණයි.",
                             "ඉගෙන ගන්නා අතරතුර පූර්ණ කාලීන රැකියා."],
            "visa_pr": ["ලකුණු පද්ධතියකින් තොරව සෘජු PR.",
                        "මාස 3කින් PR සහතිකයි.",
                        "තානාපති කාර්යාලයේ සම්බන්ධතා ඇත, {fee} ගෙවා ඉක්මනින් අනුමැතිය."],
            "local_job": ["YouTube වීඩියෝ like කර කොමිස් මුදල් ලබාගන්න.",
                          "VIP මට්ටමට {fee} ගෙවා කාර්යයන් කරන්න.",
                          "ගෙදර සිට දිනකට {salary} උපයන්න.",
                          "Telegram සමූහයට එක්වන්න, බෝනස් රු. 2,000."],
        },
        "legit": {
            "any": ["සම්මුඛ පරීක්ෂණය {day} දින අපගේ කාර්යාලයේදී.",
                    "අයදුම් කිරීමේදී කිසිදු මුදලක් ගෙවිය යුතු නැත.",
                    "මුල් සහතික පත්‍ර රැගෙන එන්න.",
                    "අපේ නමින් මුදල් ඉල්ලන වංචාකරුවන්ගෙන් පරිස්සම් වන්න."],
            "foreign_job": ["ශ්‍රී ලංකා විදේශ සේවා නියුක්ති කාර්යාංශයේ (SLBFE) බලපත්‍ර අංක {lic}.",
                            "{legit_fee} පමණක් ගෙවිය යුතුය, රිසිට්පත් නිකුත් කෙරේ.",
                            "SLBFE අනුමත රැකියා ඇණවුමකි."],
            "student_visa": ["අවම IELTS 6.0 අවශ්‍යයි.", "ප්‍රවේශ ලිපිය විශ්වවිද්‍යාලයෙන් සෘජුවම.",
                             "වීසා තීරණය තානාපති කාර්යාලය සතුය."],
            "visa_pr": ["නිල ලකුණු පද්ධතිය අනුව සුදුසුකම් තක්සේරු කෙරේ.",
                        "ලියාපදිංචි සංක්‍රමණ උපදේශක, බලපත්‍ර අංක {lic}.",
                        "කිසිදු සහතිකයක් ලබා දිය නොහැක."],
            "local_job": ["EPF/ETF ලබා දේ.", "ඔබේ CV එක මානව සම්පත් අංශයට එවන්න.",
                          "සා/පෙළ සමත් විය යුතුය."],
        },
        "contact": ["අමතන්න / WhatsApp 07X XXX XXXX.", "{agency}, දුරකථන 011 XXX XXXX.", "Inbox කරන්න.",
                    "Email: jobs@example.lk"],
        "neutral": ["WhatsApp 07X XXX XXXX.", "විස්තර සඳහා අමතන්න 07X XXX XXXX.", "පුරප්පාඩු සීමිතයි."],
        "close": {
            "scam": ["දැන්ම අමතන්න 07X XXX XXXX!!!", "Inbox කරන්න.", "Viber 07X XXX XXXX."],
            "legit": ["{agency}, දුරකථන 011 XXX XXXX.", "Email: careers@example.lk",
                      "{agency} - SLBFE බලපත්‍ර {lic}."],
        },
    },
}

# ---------------------------------------------------------------- shifted (drift) material
TAMIL = {
    "offer": {
        "foreign_job": ["{country} இல் {job} வேலை வாய்ப்பு, மாத சம்பளம் {salary}.",
                        "வெளிநாட்டு வேலை: {country} - {job} - {salary}."],
        "student_visa": ["{country} இல் {job} படிக்க வாய்ப்பு, {salary}.", "{country} மாணவர் விசா - {job}, {salary}."],
        "visa_pr": ["{country} நிரந்தர குடியுரிமை ({job}), {salary}.", "{country} {job} க்கு விண்ணப்பிக்கவும், {salary}."],
        "local_job": ["{country} இல் {job} தேவை, {salary}.", "வீட்டிலிருந்து {job} வேலை, {salary}."],
    },
    "scam": ["பதிவு கட்டணம் {fee} உடனடியாக செலுத்தவும்.", "100% விசா உத்தரவாதம்.",
             "நேர்முகத் தேர்வு இல்லை, அனுபவம் தேவையில்லை.", "பணத்தை தனிப்பட்ட கணக்கில் வைப்பு செய்யவும்.",
             "இன்றே பதிவு செய்யுங்கள், இடங்கள் குறைவு.", "கடவுச்சீட்டு நகலை WhatsApp இல் அனுப்பவும்."],
    "legit": ["SLBFE அனுமதி பெற்ற முகவர் நிலையம், உரிமம் இல. {lic}.", "முன்பணம் எதுவும் செலுத்த வேண்டியதில்லை.",
              "நேர்முகத் தேர்வு {day} அன்று எமது அலுவலகத்தில்.", "அசல் சான்றிதழ்களை கொண்டு வரவும்."],
    "contact": ["தொடர்புக்கு 07X XXX XXXX.", "மேலதிக விபரங்களுக்கு அழைக்கவும் 07X XXX XXXX.",
                "எமது அலுவலகத்திற்கு வரவும்."],
}

# ---------------------------------------------------------------- hard examples
# Hard negatives: legit ads that look suspicious (fees, WhatsApp, deadlines, deposits), so the model can't just
# learn "mentions a fee / WhatsApp / a deadline = scam". Merged into POOLS[lang]["legit"] and TAMIL["legit"].
HARD_NEGATIVES = {
    "en": {
        "any": ["Shortlisted candidates will be contacted on WhatsApp to schedule the interview.",
                "Applications close on 30 November. Late applications will not be accepted.",
                "Urgent vacancies: interviews will be held this week at our office.",
                "Interview dates are shared on WhatsApp 07X XXX XXXX after we review your CV.",
                "Limited places this intake, so apply before the closing date.",
                "Please bring a copy of your NIC and passport to the interview for verification."],
        "foreign_job": ["{agency} (SLBFE licence {lic}): the only payment is the SLBFE registration fee of Rs. 22,027, "
                        "paid at the SLBFE office with an official receipt.",
                        "Medical tests are booked at an SLBFE-approved centre and paid there, with a receipt."],
        "student_visa": ["The tuition deposit is paid only through the university's official online portal.",
                         "Seat deposit of {legit_fee} is paid directly to the university, never to an agent."],
        "visa_pr": ["Government visa fees are paid to the immigration department through its official website.",
                    "Our registered adviser charges a published consultation fee; the visa decision is the embassy's."],
        "local_job": ["Government vacancy as per the Gazette notice. An examination fee of Rs. 1,000 is payable to "
                      "the Department.",
                      "Part-time weekend role. Interview at our {country} office before selection."],
    },
    "singlish": {
        "any": ["Interview time eka WhatsApp walin kiyanawa.",
                "Applications close wenne November 30.",
                "Ikmanata apply karanna, interview me sathiye office eke.",
                "Interview ekata NIC eka passport eka copy ekak aran enna.",
                "CV eka balala WhatsApp karanawa 07X XXX XXXX."],
        "foreign_job": ["SLBFE registration fee Rs. 22,027 SLBFE office ekata gewanna, receipt ekak denawa.",
                        "Medical eka SLBFE approve karapu centre ekaka, salli ethanata gewanna."],
        "student_visa": ["University deposit eka official portal eken witharai gewanne."],
        "visa_pr": ["Visa fee eka immigration department ekata official website eken gewanna."],
        "local_job": ["Gazette eke thiyena vidihata exam fee Rs. 1,000 department ekata gewanna.",
                      "Part time job, interview ekak thiyenawa office eke."],
    },
    "si": {
        "any": ["සම්මුඛ පරීක්ෂණ වේලාව WhatsApp හරහා දැනුම් දෙනු ලැබේ.",
                "අයදුම්පත් භාරගැනීම නොවැම්බර් 30 දිනෙන් අවසන් වේ.",
                "හදිසි පුරප්පාඩු: මෙම සතියේ අපගේ කාර්යාලයේදී සම්මුඛ පරීක්ෂණ.",
                "සම්මුඛ පරීක්ෂණයට ජාතික හැඳුනුම්පතේ සහ ගමන් බලපත්‍රයේ පිටපතක් රැගෙන එන්න.",
                "ඔබේ CV එක පරීක්ෂා කළ පසු WhatsApp 07X XXX XXXX හරහා අමතනු ලැබේ."],
        "foreign_job": ["SLBFE ලියාපදිංචි ගාස්තුව රු. 22,027 SLBFE කාර්යාලයේදී පමණක් ගෙවිය යුතුය, රිසිට්පතක් ලැබේ.",
                        "වෛද්‍ය පරීක්ෂණය SLBFE අනුමත මධ්‍යස්ථානයක, ගාස්තුව එහිදීම ගෙවිය යුතුය."],
        "student_visa": ["විශ්වවිද්‍යාල තැන්පතුව නිල වෙබ් ද්වාරය හරහා පමණක් ගෙවන්න."],
        "visa_pr": ["වීසා ගාස්තුව ආගමන දෙපාර්තමේන්තුවට නිල වෙබ් අඩවිය හරහා ගෙවිය යුතුය."],
        "local_job": ["ගැසට් නිවේදනය අනුව විභාග ගාස්තුව රු. 1,000 දෙපාර්තමේන්තුවට ගෙවිය යුතුය.",
                      "අර්ධ කාලීන රැකියාව, ආයතනයේදී සම්මුඛ පරීක්ෂණයක් ඇත."],
    },
}
TAMIL["legit"] += [
    "நேர்முகத் தேர்வு நேரம் WhatsApp மூலம் அறிவிக்கப்படும்.",
    "விண்ணப்பங்கள் நவம்பர் 30 உடன் முடிவடையும்.",
    "வர்த்தமானி அறிவிப்பின்படி பரீட்சை கட்டணம் ரூ. 1,000 திணைக்களத்திற்கு செலுத்த வேண்டும்.",
    "பல்கலைக்கழக வைப்புத்தொகை உத்தியோகபூர்வ இணையதளம் மூலம் மட்டுமே செலுத்தப்படும்.",
    "பகுதி நேர வேலை, அலுவலகத்தில் நேர்முகத் தேர்வு உண்டு.",
    "SLBFE பதிவு கட்டணம் ரூ. 22,027 SLBFE அலுவலகத்தில் மட்டுமே செலுத்தப்படும்.",
]


def _merge_hard(table: dict, label: str) -> None:
    for lang, groups in table.items():
        for key, lines in groups.items():
            POOLS[lang][label].setdefault(key, []).extend(lines)


_merge_hard(HARD_NEGATIVES, "legit")

# Hard positives: scams without the usual trigger words. Polite, professional tone; fees disguised as document
# verification, medical booking or embassy appointments; "high success rate" instead of "guaranteed".
HARD_POSITIVES = {
    "en": {
        "any": ["We kindly request a refundable document verification charge of {fee} to open your file.",
                "Please book your medical through our partner centre; the medical booking fee is {fee}.",
                "An embassy appointment slot costs {fee}, payable to our coordinator.",
                "For registration, kindly send a clear copy of your NIC and passport to our officer.",
                "Our agency has a very high visa success rate with past candidates.",
                "Your offer letter will be released once the file opening charge is settled.",
                "Our coordinator will personally guide you through every step of the process."],
        "student_visa": ["Apply through us for a high visa success rate; the seat confirmation fee of {fee} is paid "
                         "to our office."],
        "visa_pr": ["We handle your PR file end to end; the assessment processing charge is {fee}."],
        "local_job": ["Paid product review internship: review items online and receive weekly payments after a "
                      "small starter deposit.",
                      "Remote product review internship for students, earn {salary}, onboarding package {fee}."],
    },
    "singlish": {
        "any": ["Document verification charge eka {fee} witharai, passe refund karanawa.",
                "Medical booking eka api karanawa, {fee} gewanna.",
                "Embassy appointment ekata {fee} coordinator ta denna.",
                "Registration walata NIC eka passport eka copy ekak yawanna.",
                "Api agency eke visa success rate eka godak high.",
                "File eka open karanna charge eka gewwama offer letter eka denawa."],
        "local_job": ["Product review internship ekak, gedara idan, starter deposit ekak witharai."],
    },
    "si": {
        "any": ["ලේඛන සත්‍යාපන ගාස්තුව {fee} පමණි, පසුව ආපසු ලබා දෙනු ලැබේ.",
                "වෛද්‍ය පරීක්ෂණය අප හරහා වෙන් කරන්න, ගාස්තුව {fee}.",
                "තානාපති කාර්යාල හමුව සඳහා {fee} අපගේ නිලධාරියාට ගෙවන්න.",
                "ලියාපදිංචිය සඳහා ජාතික හැඳුනුම්පත සහ ගමන් බලපත්‍රයේ පිටපතක් එවන්න.",
                "අපගේ වීසා සාර්ථකත්ව අනුපාතය ඉතා ඉහළයි.",
                "ගොනුව විවෘත කිරීමේ ගාස්තුව ගෙවූ පසු පත්වීම් ලිපිය ලබා දෙනු ලැබේ."],
        "local_job": ["නිෂ්පාදන සමාලෝචන අභ්‍යාසලාභී තනතුරු, ආරම්භක තැන්පතුවක් පමණි."],
    },
}
TAMIL["scam"] += [
    "ஆவண சரிபார்ப்பு கட்டணம் {fee} மட்டுமே, பின்னர் திருப்பித் தரப்படும்.",
    "மருத்துவ பரிசோதனை முன்பதிவுக்கு {fee} எம்மிடம் செலுத்தவும்.",
    "தூதரக சந்திப்புக்கு {fee} எமது ஒருங்கிணைப்பாளரிடம் செலுத்தவும்.",
    "பதிவுக்கு தேசிய அடையாள அட்டை மற்றும் கடவுச்சீட்டு நகலை அனுப்பவும்.",
    "எமது விசா வெற்றி விகிதம் மிக அதிகம்.",
    "தயாரிப்பு மதிப்பாய்வு பயிற்சி வேலை, ஆரம்ப வைப்புத்தொகை மட்டும்.",
]
_merge_hard(HARD_POSITIVES, "scam")

# Shared sign-offs and openers, used by both classes at the same rate, so licence numbers, agency names, WhatsApp
# and placeholder phones stop being label shortcuts (scams quote borrowed licences too).
SIGNOFF = {
    "en": POOLS["en"]["contact"] + POOLS["en"]["close"]["legit"] + [
        "{agency} (SLBFE licence {lic}). WhatsApp 07X XXX XXXX.",
        "Licence no. {lic} | {agency} | 07X XXX XXXX",
        "{agency}, licensed by SLBFE ({lic}). Call 011 XXX XXXX or WhatsApp 07X XXX XXXX."],
    "singlish": POOLS["singlish"]["contact"] + POOLS["singlish"]["close"]["legit"] + [
        "{agency} | licence {lic} | WhatsApp 07X XXX XXXX",
        "SLBFE licence {lic}, {agency}. Call 07X XXX XXXX."],
    "si": POOLS["si"]["contact"] + POOLS["si"]["close"]["legit"] + [
        "{agency} | බලපත්‍ර අංක {lic} | WhatsApp 07X XXX XXXX",
        "SLBFE බලපත්‍ර {lic}, {agency}. අමතන්න 07X XXX XXXX."],
}
SHARED_OPEN = {
    "en": ["Dear candidate,", "Greetings from our recruitment team.", "Dear applicants,", "Good news for job seekers."],
    "singlish": ["Ayubowan!", "Dear candidates,", "Hello hamotama,"],
    "si": ["ආයුබෝවන්!", "හිතවත් අයදුම්කරුවනි,", "රැකියා අපේක්ෂකයින්ට සුබ ආරංචියක්."],
}

NEW_TACTICS = {  # tactic -> (category, scam lines)
    "crypto_payment": ("foreign_job", [
        "Pay the processing fee in USDT (TRC20) to the wallet address below for faster visa processing.",
        "We only accept Binance Pay for the {fee} placement fee, no bank transfers.",
        "Processing fee gewanna USDT walin, wallet address eka inbox karanawa."]),
    "digital_id_fee": ("foreign_job", [
        "Link your new SL-UDI digital ID with our agency to unlock priority foreign job placement. Verification fee Rs. 4,500.",
        "Government digital ID holders get first preference. Pay the SL-UDI activation charge of Rs. 3,900 through our link.",
        "SL-UDI digital ID eka register karanna api link eka yawanawa, fee eka Rs. 4,000 witharai."]),
    "cyber_compound": ("foreign_job", [
        "High-paying customer service and IT data entry jobs in Thailand. Free air ticket, company transport arranged across the border.",
        "Typing jobs in Cambodia and Laos, USD 1,800, tourist visa enough, our team receives you at the airport.",
        "Online marketing job in Myawaddy, accommodation inside the company campus, passport kept safely by HR."]),
    "military_recruit": ("foreign_job", [
        "Security guard and logistics helper jobs in Russia, USD 2,500 monthly, contract signing in Moscow, no experience needed.",
        "Russia army support staff, Rs. 900,000 per month, citizenship after one year."]),
    "qr_payment": ("student_visa", [
        "Scan the LankaQR code to pay the seat reservation fee of {fee} for the next intake.",
        "Reserve your university seat by scanning the QR and paying {fee} within 2 hours."]),
    "social_live": ("visa_pr", [
        "Join our TikTok live tonight, visa slots announced daily for followers only.",
        "Follow our Telegram channel, AI-powered visa approval system selects 10 lucky families each week."]),
}
NEW_DESTINATIONS = {"foreign_job": ["Thailand", "Russia", "Belarus", "Georgia", "Seychelles"],
                    "student_visa": ["Uzbekistan", "Georgia", "Cyprus (North)"],
                    "visa_pr": ["Serbia", "Vanuatu"],
                    "local_job": ["online"]}


def _split_pool(items: list[str], split: str, rng: random.Random) -> list[str]:
    if len(items) < 3:
        return items
    reserved = [x for i, x in enumerate(items) if i % 3 == 2]
    common = [x for i, x in enumerate(items) if i % 3 != 2]
    if split == "train":
        return common
    return reserved if rng.random() < 0.5 else common


def _fill(template: str, rng: random.Random, cat: str, label: str, country=None, agency=None, lic=None) -> str:
    slots = SLOTS[cat]
    values = {
        "country": country or rng.choice(slots["country"]),
        "job": rng.choice(slots["job"]),
        # Salary pools overlap (30%) so the amount alone cannot give the label away.
        "salary": rng.choice(((SCAM_SALARY if label == "scam" else LEGIT_SALARY) if rng.random() < 0.7
                              else (LEGIT_SALARY if label == "scam" else SCAM_SALARY))[cat]),
        "fee": rng.choice(SCAM_FEE),
        "legit_fee": rng.choice(LEGIT_FEE),
        "agency": agency or rng.choice(AGENCY),
        "lic": lic or rng.choice(LIC),
        "day": rng.choice(DAYS),
    }
    return template.format(**values)


def _render(parts: list[str], rng: random.Random, cat: str, label: str, country=None, salary=None) -> str:
    country = country or rng.choice(SLOTS[cat]["country"])
    sep = rng.choice([" ", "\n", " ", " | "])
    # One agency per ad, used consistently in every line.
    agency = rng.choice(LEGIT_AGENCIES if label == "legit" else AGENCY)
    lic = AGENCY_LICENCE[agency] if label == "legit" or rng.random() < 0.3 else rng.choice(LIC)
    text = sep.join(_fill(p, rng, cat, label, country, agency, lic) for p in parts)
    if salary:  # shifted set: amounts written in a format the model has never seen
        for pool in (SCAM_SALARY, LEGIT_SALARY):
            for v in pool[cat]:
                text = text.replace(v, salary)
    if rng.random() < 0.3:
        text = rng.choice(["✈️ ", "📢 ", "🔥 ", "⭐ ", "🇱🇰 "]) + text
    return text


def compose(rng: random.Random, lang: str, label: str, cat: str, split: str) -> str:
    P = POOLS[lang]
    pick = lambda items: rng.choice(_split_pool(items, split, rng))  # noqa: E731
    parts = []
    r = rng.random()
    if r < 0.3:
        parts.append(pick(SHARED_OPEN[lang]))
    elif r < 0.8:
        parts.append(pick(P["open"][label]))
    parts.append(pick(P["offer"][cat]))
    generic = P[label]["any"]
    if cat == "local_job":  # visa / flight lines make no sense for a local job
        generic = [g for g in generic if not any(w in g.lower() for w in ("visa", "වීසා", "fly"))]
    pool = _split_pool(P[label][cat] + generic, split, rng)
    k = rng.randint(1, 3) if label == "scam" else rng.randint(2, 3)
    parts += rng.sample(pool, min(k, len(pool)))
    # Hard examples: most scams copy one or two lines of legit wording (deadlines, interview scheduling, document
    # checks), so no single phrase gives the class away; legit ads carry "risky-looking" but harmless lines.
    if label == "scam" and rng.random() < 0.6:
        borrowed = _split_pool(P["legit"]["any"], split, rng)
        for line in rng.sample(borrowed, rng.randint(1, 2)):
            parts.insert(rng.randint(2, len(parts)), line)
    if rng.random() < 0.45:  # same rate for both classes, so WhatsApp / phone lines stay balanced
        parts.append(pick(P["neutral"]))
    # Scam-only closes are urgency tactics (a real signal); the sign-off itself is shared by both classes.
    if label == "scam" and rng.random() < 0.25:
        parts.append(pick(P["close"]["scam"]))
    parts.append(pick(SIGNOFF[lang]))
    return _render(parts, rng, cat, label)


def generate(n: int, split: str, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    langs, weights = ["en", "singlish", "si"], [0.5, 0.3, 0.2]
    rows = []
    for _ in range(n):
        lang = rng.choices(langs, weights)[0]
        cat = rng.choice(CATEGORIES)
        label = "scam" if rng.random() < 0.45 else "legit"
        rows.append({"text": compose(rng, lang, label, cat, split), "label": label, "category": cat,
                     "language": lang, "source": "synthetic", "tactic": "baseline"})
    return pd.DataFrame(rows).drop_duplicates("text")


def generate_shifted(n: int, seed: int) -> pd.DataFrame:
    """Drift set: ~40% Tamil, ~40% new scam tactics / destinations, ~20% ordinary in-distribution offers."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r = rng.random()
        if r < 0.4:  # new language
            cat = rng.choice(CATEGORIES)
            label = "scam" if rng.random() < 0.5 else "legit"
            parts = [rng.choice(TAMIL["offer"][cat])]
            parts += rng.sample(TAMIL[label], rng.randint(2, 3))
            parts.append(rng.choice(TAMIL["contact"]))
            salary = rng.choice(["மாதம் 5 லட்சம் ரூபாய்", "மாதம் 1.5 லட்சம் ரூபாய்", "மாதம் 3 லட்சம் ரூபாய்",
                                 "நாளொன்றுக்கு 8 ஆயிரம் ரூபாய்", "மாதம் 90 ஆயிரம் ரூபாய்"])
            text, lang, tactic = _render(parts, rng, cat, label, salary=salary), "ta", "tamil"
        elif r < 0.8:  # new tactics and destinations
            tactic = rng.choice(list(NEW_TACTICS))
            cat, lines = NEW_TACTICS[tactic]
            label = "scam" if rng.random() < 0.8 else "legit"
            lang = rng.choice(["en", "en", "singlish"])
            country = rng.choice(NEW_DESTINATIONS[cat])
            if label == "scam":
                parts = [rng.choice(POOLS[lang]["offer"][cat]), rng.choice(lines)]
                if rng.random() < 0.5:
                    parts.append(rng.choice(POOLS[lang]["neutral"]))
                parts.append(rng.choice(["Telegram: @newjobs_lk", "TikTok @visa.fast.lk", "DM for the wallet address."]))
            else:  # legit offers to the new destinations, written the official way
                parts = [rng.choice(POOLS[lang]["offer"][cat])]
                parts += rng.sample(POOLS[lang]["legit"][cat] + POOLS[lang]["legit"]["any"], 2)
                parts.append(rng.choice(POOLS[lang]["close"]["legit"]))
            salary = rng.choice(["USD 2,200", "THB 35,000", "RUB 180,000", "USD 1,650", "EUR 1,300", "GEL 2,000"])
            text = _render(parts, rng, cat, label, country=country, salary=salary)
        else:  # ordinary traffic
            lang = rng.choices(["en", "singlish", "si"], [0.5, 0.3, 0.2])[0]
            cat = rng.choice(CATEGORIES)
            label = "scam" if rng.random() < 0.45 else "legit"
            text, tactic = compose(rng, lang, label, cat, "test"), "baseline"
        rows.append({"text": text, "label": label, "category": cat, "language": lang,
                     "source": "synthetic_shifted", "tactic": tactic})
    return pd.DataFrame(rows).drop_duplicates("text")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-shifted", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = generate(args.n_train, "train", args.seed)
    test = generate(args.n_test, "test", args.seed + 1)
    test = test[~test["text"].isin(train["text"])]  # no exact duplicates across splits
    shifted = generate_shifted(args.n_shifted, args.seed + 2)
    shifted = shifted[~shifted["text"].isin(train["text"])]

    for name, df in [("synthetic_train", train), ("synthetic_test", test), ("shifted", shifted)]:
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows | labels {df['label'].value_counts().to_dict()} | "
              f"langs {df['language'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
