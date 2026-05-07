# TODO - AI Orchestration HW1

## Phase 1 - Planning

- [x] Create GitHub repository.
- [x] Clone repository to local computer.
- [x] Create `prd.md`.
- [x] Create `plan.md`.
- [x] Create `todo.md`.
- [ ] Verify that every PRD requirement appears in this TODO list.
- [x] Create first Git commit for planning documents.

---

## Phase 2 - Project Structure

- [x] Create `.gitignore`.
- [x] Create `requirements.txt`.
- [x] Create `main.py`.
- [x] Create `src/` folder.
- [x] Create `tests/` folder.
- [x] Create `results/` folder.
- [x] Create `src/data.py`.
- [x] Create `src/models.py`.
- [x] Create `src/train.py`.
- [x] Create `src/evaluate.py`.
- [x] Create `tests/test_data.py`.
- [x] Create `tests/test_models.py`.

---

## Phase 3 - Dataset

- [x] Choose 4 known frequencies.
- [x] Explain the frequency choices in README.
- [x] Generate clean sine/cosine signals.
- [x] Add random noise using `sigma`.
- [x] Create 10-sample noisy windows.
- [x] Create matching 10-sample clean target windows.
- [x] Create one-hot encoded frequency vector.
- [x] Return dataset in PyTorch-friendly format.
- [x] Test dataset output shapes.
- [x] Test one-hot encoding validity.
- [x] Test that noise changes the clean signal.

---

## Phase 4 - Models

- [x] Implement MLP model.
- [x] Implement RNN model.
- [x] Implement LSTM model.
- [x] Test MLP forward pass.
- [x] Test RNN forward pass.
- [x] Test LSTM forward pass.
- [x] Test that all models output 10 reconstructed samples.

---

## Phase 5 - Training

- [x] Implement MSE loss.
- [x] Implement Adam optimizer.
- [x] Implement training loop.
- [x] Train MLP.
- [x] Train RNN.
- [x] Train LSTM.
- [x] Record training losses.
- [x] Split dataset into train and test sets.

---

## Phase 6 - Evaluation

- [x] Calculate test MSE for MLP.
- [x] Calculate test MSE for RNN.
- [x] Calculate test MSE for LSTM.
- [x] Compare the three models.
- [x] Save at least one prediction plot.
- [ ] Save loss/results in `results/`.
- [x] Discuss which model performed best.

---

## Phase 7 - README Lab Report

- [x] Write project overview.
- [x] Explain RNN and sequence prediction.
- [x] Explain sine signal, frequency, amplitude, phase, and sampling.
- [x] Explain dataset generation.
- [x] Explain model architectures.
- [x] Explain training setup.
- [x] Add results table.
- [x] Add discussion.
- [x] Add instructions for running the project.
- [x] Add instructions for running tests.
- [x] Add GitHub repository link.

---

## Phase 8 - Final Checks

- [x] Run `python main.py`.
- [x] Run `pytest`.
- [x] Fix all errors.
- [x] Check that files are modular and not too long.
- [x] Check that README is detailed enough.
- [ ] Check that GitHub repository is public.
- [ ] Push final version to GitHub.
- [ ] Prepare final PDF/report with GitHub link.