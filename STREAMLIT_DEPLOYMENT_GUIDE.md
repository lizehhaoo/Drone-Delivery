# Streamlit Live Demo Deployment Guide

1. Create a public GitHub repository.
2. Upload these files at the root of the repository. Do not keep the `(14)`, `(5)`, or `(11)` suffixes in filenames.
3. Confirm the repository root contains:
   - `interactive_model_app.py`
   - `payload_drone_backend_interactive_fixed_only.py`
   - `manitoba_household_candidates_ALL_points.csv`
   - `requirements.txt`
4. Go to Streamlit Community Cloud and create a new app from your GitHub repo.
5. Use `interactive_model_app.py` as the main file path.
6. After deployment, copy the Streamlit URL and replace the placeholder Live Demo link in `README.md`.

Local test command:

```bash
pip install -r requirements.txt
streamlit run interactive_model_app.py
```
