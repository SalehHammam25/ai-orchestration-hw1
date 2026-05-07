# TODO - AI Orchestration HW1

## Phase 1 - Planning

- [x] Create GitHub repository.
- [x] Clone repository to local computer.
- [x] Create `prd.md`.
- [x] Create `plan.md`.
- [x] Create `todo.md`.
- [ ] Verify that every PRD requirement appears in this TODO list.
- [ ] Create first Git commit for planning documents.

---

## Phase 2 - Project Structure

- [ ] Create `.gitignore`.
- [ ] Create `requirements.txt`.
- [ ] Create `main.py`.
- [ ] Create `src/` folder.
- [ ] Create `tests/` folder.
- [ ] Create `results/` folder.
- [ ] Create `src/data.py`.
- [ ] Create `src/models.py`.
- [ ] Create `src/train.py`.
- [ ] Create `src/evaluate.py`.
- [ ] Create `tests/test_data.py`.
- [ ] Create `tests/test_models.py`.

---

## Phase 3 - Dataset

- [ ] Choose 4 known frequencies.
- [ ] Explain the frequency choices in README.
- [ ] Generate clean sine/cosine signals.
- [ ] Add random noise using `sigma`.
- [ ] Create 10-sample noisy windows.
- [ ] Create matching 10-sample clean target windows.
- [ ] Create one-hot encoded frequency vector.
- [ ] Return dataset in PyTorch-friendly format.
- [ ] Test dataset output shapes.
- [ ] Test one-hot encoding validity.
- [ ] Test that noise changes the clean signal.

---

## Phase 4 - Models

- [ ] Implement MLP model.
- [ ] Implement RNN model.
- [ ] Implement LSTM model.
- [ ] Test MLP forward pass.
- [ ] Test RNN forward pass.
- [ ] Test LSTM forward pass.
- [ ] Test that all models output 10 reconstructed samples.

---

## Phase 5 - Training

- [ ] Implement MSE loss.
- [ ] Implement Adam optimizer.
- [ ] Implement training loop.
- [ ] Train MLP.
- [ ] Train RNN.
- [ ] Train LSTM.
- [ ] Record training losses.
- [ ] Split dataset into train and test sets.

---

## Phase 6 - Evaluation

- [ ] Calculate test MSE for MLP.
- [ ] Calculate test MSE for RNN.
- [ ] Calculate test MSE for LSTM.
- [ ] Compare the three models.
- [ ] Save at least one prediction plot.
- [ ] Save loss/results in `results/`.
- [ ] Discuss which model performed best.

---

## Phase 7 - README Lab Report

- [ ] Write project overview.
- [ ] Explain RNN and sequence prediction.
- [ ] Explain sine signal, frequency, amplitude, phase, and sampling.
- [ ] Explain dataset generation.
- [ ] Explain model architectures.
- [ ] Explain training setup.
- [ ] Add results table.
- [ ] Add discussion.
- [ ] Add instructions for running the project.
- [ ] Add instructions for running tests.
- [ ] Add GitHub repository link.

---

## Phase 8 - Final Checks

- [ ] Run `python main.py`.
- [ ] Run `pytest`.
- [ ] Fix all errors.
- [ ] Check that files are modular and not too long.
- [ ] Check that README is detailed enough.
- [ ] Check that GitHub repository is public.
- [ ] Push final version to GitHub.
- [ ] Prepare final PDF/report with GitHub link.