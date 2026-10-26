# BITS Pilani Machine Learning Assignment 2 - Lab-to-submission guide

**Deadline:** Sunday, 8 November 2026, 11:59 PM (assignment states 23:59).
**Marks:** 15. **Output:** ONE PDF (GitHub link, live Streamlit link,
one BITS Virtual Lab screenshot, and full README content, IN THAT ORDER).
**Requirement:** Complete the actual work inside BITS Virtual Lab.
**Integrity:** AI support is permitted for learning, NOT unedited copy-paste submission.

## Step 1 - Start from BITS Virtual Lab

1. Log in to BITS Virtual Lab and open the Python terminal or development environment.
2. Upload or clone this guided project folder into Virtual Lab.
3. Verify Python and install dependencies (adapt to the lab's shell):

```bash
python --version
python -m pip install -r requirements.txt
```

If your lab is using a supported virtual environment, activate it first.
Use Python 3.12 when practical for Streamlit Community Cloud parity.

## Step 2 - Download ONE eligible UCI dataset

- UCI #468: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- 12,330 rows; 17 original features; binary target Revenue.
- Download the file and keep the attribution to UCI.

```bash
python download_dataset.py
```

If downloading from the lab terminal is blocked, download
`online_shoppers_intention.csv` from the UCI webpage via your browser and put it
inside `data/` manually. Confirm `data/online_shoppers_intention.csv` exists.

## Step 3 - Study what the models do and pick your settings

Read `common.py` and `train_models.py` before running them.

- The target is Revenue: purchase=1, not purchase=0.
- Split original data ONCE into training and untouched test partitions (80/20).
- Stratification preserves the minority class distribution.
- Fit encoding, imputation and scaling on training data only.
- Save preprocessing + estimator in one joblib Pipeline.
- Train the five required models, independently, on the same training set.
- Evaluate all models on exactly the same test set.
- ROC AUC is calculated from continuous scores, not hard predictions.
- Describe your hyperparameter choices and own observations.

## Step 4 - Execute model training on BITS Virtual Lab

```bash
python train_models.py
```

Expected files after training:

- `test_data.csv` (CSV: 17 features + Revenue target)
- `model_comparison.csv` (all six metrics x five models)
- `model/logistic_regression.joblib`
- `model/decision_tree.joblib`
- `model/knn.joblib`
- `model/naive_bayes.joblib`
- `model/support_vector_machine.joblib`

You should see five printed result rows. Record your OWN conclusions rather
than adopting a predetermined "winner." The dataset is imbalanced, so AUC,
F1, MCC and minority-class Recall may matter more than headline Accuracy.

## Step 5 - Run the interactive Streamlit app in BITS Lab

```bash
streamlit run app.py
```

In the lab browser open the URL printed by Streamlit, often `http://localhost:8501`.
The interface must show:

- CSV uploader for LABELLED test data, including Revenue.
- Model selection dropdown with five classifiers.
- Six metrics: Accuracy, AUC, Precision, Recall, F1, MCC.
- Confusion matrix for the selected classifier.
- All five model results when using test_data.csv.

Upload the generated `test_data.csv` to demonstrate the upload feature.
Take ONE clear screenshot INSIDE BITS Virtual Lab showing the app, metrics
and the lab context. Save it as `evidence/bits_virtual_lab.png`.

## Step 6 - Build your GitHub repo

1. Create a NEW GitHub repository under YOUR account, not a fork of someone
   else's assignment repository. Example: `shopping-intent-ml-bits-a2`.
2. Add all project files and resulting model/*.joblib and test_data.csv.
3. Include requirements.txt and the final README.md.
4. Make commits as you work, with meaningful descriptions. Commit history
   is reviewed for academic integrity by the assignment graders.
5. NEVER commit credentials, tokens or secrets.

Suggested commands if your BITS Lab permits GitHub access:

```bash
git init
git add .
git commit -m "Study: preprocessing and model comparison"
git branch -M main
git remote add origin https://github.com/YOUR-USER/YOUR-REPO.git
git push -u origin main
```

Create the GitHub repository first and fill in the real YOUR-USER/YOUR-REPO.
If your BITS lab blocks Git operations, manually transfer your own files
and upload them using GitHub's web UI. Keep real commit history.

## Step 7 - Deploy to Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Click Create app (wording may differ from the PDF's "New App").
3. Select the GitHub repository, main branch and app.py.
4. Use Python 3.12 if given the option; dependencies are in requirements.txt.
5. Deploy and wait for the app to open.
6. On the LIVE link upload test_data.csv, choose all five models and inspect
   evaluation metrics and confusion matrix. Confirm nothing retrains.
7. Copy your real `https://...streamlit.app` URL for the submission.

Official deployment guide:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Step 8 - Write YOUR observations from actual metrics

Open `model_comparison.csv` and study the results. For each model, note at
least one comparative strength and limitation supported by YOUR data:

- Logistic Regression: does its Recall match its Precision?
- Decision Tree: does interpretable splitting perform better or worse?
- KNN: how well does a neighbor-based approach identify purchases?
- Naive Bayes: does the conditional independence assumption limit results?
- SVM: is AUC or MCC better than the simpler alternatives?

Choose an overall winner, justifying it using your own metric priorities.
Do not assume the model with the highest Accuracy is best on imbalanced data.

## Step 9 - Generate your one-PDF submission

Edit `submission_details.json`:

- Student name and student ID.
- Actual GitHub repository URL.
- Actual deployed Streamlit URL.
- One REAL lab screenshot path (`evidence/bits_virtual_lab.png`).
- FIVE observations based on your real scores.
- One justified overall winning classifier.

After saving:

```bash
python prepare_submission.py
```

It fills in the README with real metrics and YOUR supplied observations,
then creates `submission.pdf` containing, in order:

1. Clickable GitHub repository URL.
2. Clickable live Streamlit URL.
3. The genuine BITS Lab execution screenshot.
4. The complete current README text and tables.

Review the generated PDF. Push the updated README.md to GitHub after this
step, then confirm both links still work.

## Step 10 - Final check in BITS submission portal

- [ ] Dataset >= 500 rows and >= 12 features.
- [ ] Same dataset / holdout data used for all five required models.
- [ ] Accurate Accuracy, AUC, Precision, Recall, F1 and MCC for each model.
- [ ] 5 saved trained Pipelines in model/.
- [ ] test_data.csv includes the target Revenue column.
- [ ] App opens on Streamlit Community Cloud and evaluates new test CSV WITHOUT retraining.
- [ ] GitHub includes source, dependency file, clear README, test CSV, model files.
- [ ] One REAL screenshot shows execution inside BITS Virtual Lab.
- [ ] README is included in the submitted PDF.
- [ ] Links in PDF are clickable and tested.
- [ ] Student-written observations and a justified winner are included.
- [ ] Upload ONE final PDF (not a draft), and press SUBMIT by 08-Nov-2026 23:59.

**BITS lab assistance (from assignment PDF):**
csislabsupport@wilp.bits-pilani.ac.in and raj.rohan@wilp.bits-pilani.ac.in
Subject: `NSP1 ML Assignment 2: BITS Lab issue`
