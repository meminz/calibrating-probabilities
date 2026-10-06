# Calibration of Probabilities

## repo structure

src
├── __init__.py
├── data.py                 -- datasets loader
├── models.py               -- base models
├── platt_scaling.py        -- Platt scaler class
└── prediction.py           -- out-of-sample protocol
scripts
├── __init__.py
├── fit_platt.py            -- Platt scaling fitting
└── train_base_models.py    -- launch training of models
main.py                     -- entry point
