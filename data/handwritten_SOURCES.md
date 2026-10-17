# Handwritten real-world test set: sources

`data/handwritten.csv` holds real Sri Lankan job and visa ads, collected by hand on 2026-10-17. It is a **test-only** set. Never train, tune thresholds or pick models on it, and never add it to any training split.

## Collection rules

- Only public pages that need no login: the SLBFE Online Job Bank, public ikman.lk ad pages (listing pages and ad pages only, no query-parameter URLs, as robots.txt requires), official notices, and news or fact-check articles.
- No Facebook groups, WhatsApp, Telegram channels or other closed or private sources (competition Rule 2.5).
- **Legit rows** are job orders copied from the SLBFE Online Job Bank and rendered as the text a job seeker sees, real ikman.lk ads (preferring 'AUTH AGENT' licensed agencies and well-known employers), and official scholarship and embassy notices.
- **Scam rows**: news and official warnings rarely quote a scam ad word for word. Most scam rows are therefore short, faithful reconstructions of the offer each article describes. Their `label_basis` says "reconstructed from description in ..." and they are marked `needs_review=yes`. Two scam rows are real ikman.lk ads that only *look* like the work-from-home task-job pattern the Police warn about. They are marked as suspected and need a human decision.
- Sinhala and Tamil reconstructions were written for this set from the described offer; they are not quotes.
- Copyright: only the short ad wording is used. No article text is reproduced.

## Personal data removal (Rule 2.4, Sri Lanka PDPA)

Every phone number, email, personal name, street address and personal link was replaced with `<PHONE>`, `<EMAIL>`, `<NAME>`, `<ADDRESS>` and `<URL>`. Names of small private employers and organisations were replaced with `<COMPANY>` or `<ORGANISATION>`. Agency and company names are kept only in legit rows from SLBFE-licensed agencies, large listed companies or official bodies. In scam rows every agency name is `<AGENCY>`. `tests/test_handwritten.py` checks that no text matches Sri Lankan phone, email or NIC patterns.

## Rows

Row N is the Nth data record, not counting the header. Some texts span several lines, so row numbers do not match file line numbers.

| Row | Label | Source URL | Accessed | Removed / replaced |
|---|---|---|---|---|
| 1 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3532 | 2026-10-17 | nothing (no personal data in source) |
| 2 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3532 | 2026-10-17 | nothing (no personal data in source) |
| 3 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3532 | 2026-10-17 | nothing (no personal data in source) |
| 4 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3532 | 2026-10-17 | nothing (no personal data in source) |
| 5 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3532 | 2026-10-17 | nothing (no personal data in source) |
| 6 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3532 | 2026-10-17 | nothing (no personal data in source) |
| 7 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3100 | 2026-10-17 | nothing (no personal data in source) |
| 8 | legit | https://applications.slbfe.lk/jobbank/jsearchdisplay_an_m.asp?an=3100 | 2026-10-17 | nothing (no personal data in source) |
| 9 | legit | https://ikman.lk/en/ad/mechanics-serbia-colombo | 2026-10-17 | phone numbers, emails, street address |
| 10 | legit | https://ikman.lk/en/ad/bike-delivery-riders-kuwait-colombo-1 | 2026-10-17 | phone numbers, street address |
| 11 | legit | https://ikman.lk/en/ad/office-admin-female-maldives-colombo-3 | 2026-10-17 | phone numbers, emails, street address |
| 12 | legit | https://ikman.lk/en/ad/heavy-vehicle-driver-kuwait-colombo-1 | 2026-10-17 | phone numbers, emails, street address |
| 13 | legit | https://ikman.lk/en/ad/housemaid-kuwait-kurunegala-2 | 2026-10-17 | phone numbers, emails, street address, website/link |
| 14 | legit | https://ikman.lk/en/ad/allevi-vidhaayk-kurunnaaegl-kurunegala | 2026-10-17 | phone numbers |
| 15 | legit | https://ikman.lk/en/ad/bedaahaeriimee-nildhaari-pirimi-gaehaennu-colombo-10 | 2026-10-17 | phone numbers |
| 16 | legit | https://ikman.lk/en/ad/driver-battaramulla-colombo-23 | 2026-10-17 | nothing (no personal data in source) |
| 17 | legit | https://ikman.lk/en/ad/front-office-officer-colombo | 2026-10-17 | phone numbers, emails, street address, employer name |
| 18 | legit | https://ikman.lk/en/ad/aemtum-mdhysthaan-nildhaarii-nny-aykiriim-colombo | 2026-10-17 | phone numbers, street address, employer name |
| 19 | legit | https://ikman.lk/en/ad/gbddaa-shaaykyin-stores-helpers-kdduvel-colombo | 2026-10-17 | phone numbers, emails |
| 20 | legit | https://ikman.lk/en/ad/pastry-maker-jaffna-1 | 2026-10-17 | phone numbers |
| 21 | legit | https://ikman.lk/en/ad/office-manager-jaffna-jaffna | 2026-10-17 | phone numbers, organisation name |
| 22 | legit | https://ikman.lk/en/ad/virrpnnnaiyaallr-manipay-jaffna | 2026-10-17 | phone numbers |
| 23 | legit | https://ikman.lk/en/ad/nurse-chavakachcheri-jaffna | 2026-10-17 | phone numbers |
| 24 | legit | https://ikman.lk/en/ad/store-keeper-jaffna-jaffna-1 | 2026-10-17 | nothing (no personal data in source) |
| 25 | legit | https://ikman.lk/en/ad/supervisor-gampaha-gampaha-119 | 2026-10-17 | phone numbers, emails, personal names |
| 26 | legit | https://ikman.lk/en/ad/head-chef-batticaloa-2 | 2026-10-17 | emails |
| 27 | legit | https://uom.lk/eap/news/japanese-government-mext-scholarships-2027-ministry-education-higher-education-and | 2026-10-17 | nothing (no personal data in source) |
| 28 | legit | https://www.sundayobserver.lk/2024/02/04/news-features/15811/indian-schols-for-sri-lankans | 2026-10-17 | phone numbers |
| 29 | legit | https://adaderana.lk/news/105566/us-embassy-issues-special-notice-for-visa-applicants | 2026-10-17 | nothing (no personal data in source) |
| 30 | legit | https://hirunews.lk/english/business/484299/korean-embassy-introduces-visa-application-system-in-sri-lanka | 2026-10-17 | nothing (no personal data in source) |
| 31 | scam | https://www.dailymirror.lk/recomended-news/Fake-website-misuses-Foreign-Bureau-name-to-collect-job-seekers-personal-data/277-351079 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 32 | scam | https://www.dailymirror.lk/recomended-news/Fake-website-misuses-Foreign-Bureau-name-to-collect-job-seekers-personal-data/277-351079 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 33 | scam | https://www.adaderana.lk/news/104323/large-scale-foreign-employment-scam-uncovered- | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 34 | scam | https://www.slbfe.lk/?p=3200 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 35 | scam | https://www.slbfe.lk/?p=3093 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 36 | scam | https://www.adaderana.lk/news/111803/suspect-arrested-for-defrauding-over-rs-150-mln-by-promising-jobs-in-korea- | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 37 | scam | https://hirunews.lk/en/485236/hiru-cia-uncovers-large-scale-fraud-using-canadian-work-visas | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 38 | scam | https://www.dailymirror.lk/breaking-news/Three-arrested-over-Rs-30-million-Canada-job-scam/108-337737 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers, agency name |
| 39 | scam | https://www.adaderana.lk/news/114876/two-arrested-over-foreign-employment-scams | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 40 | scam | https://www.ft.lk/front-page/SLBFE-busts-major-foreign-job-scandals-over-Rs-21-49-m/44-782871 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 41 | scam | https://www.ft.lk/front-page/SLBFE-busts-major-foreign-job-scandals-over-Rs-21-49-m/44-782871 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers, agency name |
| 42 | scam | https://www.ft.lk/front-page/SLBFE-busts-major-foreign-job-scandals-over-Rs-21-49-m/44-782871 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 43 | scam | https://www.adaderana.lk/news/118697/three-including-woman-arrested-over-separate-foreign-employment-scams | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers, agency name |
| 44 | scam | https://www.adaderana.lk/news/118697/three-including-woman-arrested-over-separate-foreign-employment-scams | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 45 | scam | https://www.slbfe.lk/?p=3422 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 46 | scam | https://www.adaderana.lk/news/117814/slbfe-warns-of-overseas-based-job-scam-promising-employment-in-moldova | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 47 | scam | https://www.kyivpost.com/post/34081 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 48 | scam | https://english.newsfirst.lk/2023/12/28/sri-lankans-trapped-in-myanmar-cyber-slavery-unraveling-the-chinese-fueled-racket | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 49 | scam | http://www.newswire.lk/2023/12/31/rise-in-online-job-scams-via-telegram-app-police/ | 2026-10-17 | website/link |
| 50 | scam | https://www.dailymirror.lk/breaking-news/Rise-in-social-media-scams-targeting-Sri-Lankans-CID-warns-public/108-312761 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 51 | scam | https://www.dailymirror.lk/breaking-news/Rise-in-social-media-scams-targeting-Sri-Lankans-CID-warns-public/108-312761 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: none |
| 52 | scam | https://adaderana.lk/news/105738/public-warned-of-fake-job-scams-misusing-cbsl-logo | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 53 | scam | https://english.factcrescendo.com/consumer-safety-fact-check/fake-job-vacancies-advertised-for-srilankan-airlines-via-a-fraudulent-blog-site-619682/ | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 54 | scam | https://adaderana.lk/news.php?nid=105572 | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 55 | scam | https://www.canberratimes.com.au/story/8173149/fake-agents-dupe-aspiring-south-asian-student-migrants/ | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 56 | scam | https://www.canberratimes.com.au/story/8173149/fake-agents-dupe-aspiring-south-asian-student-migrants/ | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 57 | scam | https://www.newsfirst.lk/2018/08/02/new-zealand-student-visa-scam-in-sri-lanka | 2026-10-17 | nothing to remove (reconstruction); placeholders used: phone numbers |
| 58 | scam | https://peopledaily.digital/news/us-department-of-state-shares-2027-green-card-advisory | 2026-10-17 | nothing to remove (reconstruction); placeholders used: website/link |
| 59 | scam | https://ikman.lk/en/ad/data-entry-operators-part-time-works-from-home-badulla-5 | 2026-10-17 | nothing (no personal data in source) |
| 60 | scam | https://ikman.lk/en/ad/computer-type-setter-batticaloa-batticaloa | 2026-10-17 | website/link |

## Counts

- Rows: 60
- Label: legit: 30, scam: 30
- Category: foreign_job: 30, local_job: 21, student_visa: 4, visa_pr: 5
- Language: en: 39, mixed: 3, si: 10, singlish: 3, ta: 5
- needs_review: no: 19, yes: 41

A teammate must check every label before any number from this set is quoted.
