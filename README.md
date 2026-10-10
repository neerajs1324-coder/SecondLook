# Second Look

A scam text checker. Paste a suspicious text message and it tells you how likely it is to be a scam, **why**, and what to do about it. The model runs entirely in the browser, so messages never leave your device.

**Live site:** https://secondlook-web-phi.vercel.app

Americans lost about $21 billion to online crime in 2025, and a lot of it starts with a single fake text: "your package is on hold," "your account is locked," "you owe a toll." Second Look is built for the people who get hit hardest, like older relatives, so the results are written in plain English with large, readable type.

## What it does

- **Scam estimate** from a machine learning model trained on about 5,000 real labeled text messages
- **Plain English reasons**, like "the link uses a company's name but isn't that company's real website"
- **Highlights** the exact words and links that raised the score
- **Next steps** for each result, like forwarding scams to 7726
- **Private by design:** no server, no tracking, nothing is uploaded

## How it works

**1. Cleaning the data.** Training data comes from two public research datasets ([details below](#data)). An audit showed most of the 2022 dataset is copied from the 2011 one, often with broken text encoding that hides the match. I repaired encoding, removed exact duplicates using a letters and digits only key, and merged the two sources (6,006 unique messages).

**2. Splitting honestly.** Scam texts come in campaigns: the same template with different numbers. Clustering near duplicates with MinHash showed **62% of scam messages belong to a multi message campaign**, vs 7% of normal texts. A random split would test the model on campaigns it already saw and inflate the score, so I split **by campaign** with `StratifiedGroupKFold` (train / validation / test, about 70 / 15 / 15).

**3. Reading a message.** Each message becomes word and two word phrases, with links, phone numbers, dollar amounts, and other numbers replaced by placeholder tokens so specific digits never become vocabulary. Then 16 red flag checks run: lookalike brand domains (`usps-redelivery.top`), link shorteners, sketchy web endings like `.xyz`, raw IP links, urgency, prizes, gift card or crypto payment, requests for codes or logins, package problems, locked accounts, and unpaid tolls.

**4. The model.** Multinomial logistic regression over TF-IDF features plus the red flags, with class balancing and regularization tuned on the validation set. Logistic regression was chosen on purpose: every word has a weight you can inspect, which is what powers the "why" explanations.

**5. Running in the browser.** The trained weights are exported to JSON and `web/scorer.js` reimplements the feature extraction line for line. A parity test runs 883 messages through both versions and checks they match (largest difference: 0.0000005).

**6. Safety net.** If a serious warning sign shows up, like a fake brand domain, the site says "be careful" even when the wording alone looks normal.

## Results

Held out test set of 851 messages (709 normal, 62 spam, 80 scam), never used for training or tuning:

| Model | Macro F1 | Scams caught | Normal texts called scam |
|---|---|---|---|
| Naive Bayes, words only | 0.765 | 75% | 0.0% |
| Red flags only | 0.715 | 86% | 1.1% |
| Logistic regression, words only | 0.892 | 91% | 0.1% |
| **Logistic regression, words + red flags** | **0.882** | **90%** | **0.3%** |

With the website's full decision rule, it **warns about 91% of scams** and wrongly warns about **0.6% of normal texts**.

**Why keep the red flags if they don't help on the test set?** The test set is mostly older UK texts. On 32 hand written 2026 style examples (USPS, tolls, bank alerts, fake refunds, plus normal texts like verification codes and delivery updates), the words only model called 11 of 16 scams a scam and falsely flagged 1 normal text. Adding red flags raised that to **13 of 16 with no false alarms**, and the full website rule warns on 14 of 16. These examples were written for this project to sanity check newer scams. They aren't real collected messages and aren't counted in the numbers above. See `reports/metrics.json`.

## Limitations

- **Conversational scams with no link**, like "Hi Mom, this is my new number" or fake job offers, usually get missed. They look like normal texts until later in the conversation.
- **The training data is old.** Most of it is from around 2011 to 2022, so the red flag checks carry a lot of the weight on newer scams.
- **Built for SMS.** Long emails are a different style of writing and may score less accurately.
- This is a learning project, not a security product.

## Run it

```
pip install -r requirements.txt
python python/build_dataset.py     # clean, dedupe, and split the data
python python/train.py             # train, evaluate, export web/model.json
pytest tests/                      # red flag rule tests
node tests/check_parity.mjs        # browser and Python give identical scores
cd web && python -m http.server    # then open http://localhost:8000
```

To deploy on Vercel, import the repo and set the **Root Directory** to `web`.

## Project layout

```
python/features.py        tokenizing and red flag checks
python/build_dataset.py   encoding repair, dedupe, campaign clustering, split
python/train.py           training, baselines, evaluation, export
web/index.html            the website
web/scorer.js             browser port of the features and model
web/rules.json            red flag rules shared by Python and the browser
web/model.json            exported model weights
examples/                 hand written 2026 style test messages
reports/metrics.json      full evaluation results
data/                     raw datasets, cleaned corpus, data audit
```

## Data

- **SMS Phishing Dataset for Machine Learning and Pattern Recognition**, Mishra and Soni, 2022. [Mendeley Data](https://data.mendeley.com/datasets/f45bkkt8pr/1), CC BY 4.0.
- **SMS Spam Collection**, Almeida and Gómez Hidalgo, 2011. [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), CC BY 4.0.
