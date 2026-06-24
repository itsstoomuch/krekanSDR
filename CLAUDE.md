# antiJAMsimulation Project Memory

## What this project is
Python simulation of a hybrid analog-digital GPS anti-jamming system.
4-element antenna array. Pure math on synthetic IQ data. No hardware yet.

## Stack
- Python 3
- NumPy (math)
- Matplotlib (plots)
- SciPy (eigenvalue decomposition for MUSIC)

## Physical parameters
- GPS L1 frequency: 1575.42 MHz
- Antenna spacing: half-wavelength = 9.5 cm
- Number of antennas: 4 elements
- Jammer angle: 45 degrees
- Jammer power: 30 dB above GPS
- IQ samples per channel: 1000

## The 4 simulations to build
1. generate_array_data.py — 4-channel synthetic IQ generator
2. music_spectrum.py — MUSIC algorithm angle estimation
3. mvdr_weights.py — MVDR null steering beamformer
4. hybrid_sim.py — hybrid analog pre-cancel + digital MVDR
5. run_all.py — master script, calls all four, generates all plots

## Novel contribution
Hybrid analog-digital beamformer. Vector modulator does coarse
analog cancellation before ADC. Extends dynamic range compared
to pure digital MVDR. Key result: SNR vs jammer power curve
showing hybrid stays higher longer before degrading.

## Style rules
- One function per file
- Every function has a clear docstring
- All plots publication quality
- Comments explain the physics not just the code


## Current Status — All Fixes Applied

FIXES APPLIED:
1. Reduced to 2 jammers in generate_array_data.py (jammer3 removed)
2. Added normalisation in music_spectrum.py: X = X / max(abs(X))
3. Added diagonal loading: R = R + 1e-6 * eye(4)
4. n_signals = 3 (GPS + 2 jammers)
5. Full azimuth scan -180 to +180

VERIFIED RESULTS:
- Jammer 1 detected at +30.95° (error 0.01°)
- Jammer 2 detected at +165.95° (error 0.01°)
- MVDR null depths: 68 dB (J1), 56 dB (J2)
- Hybrid extends range by 14 dB over pure digital
- All COGNAV-4 targets PASS

FILES COMPLETED:
- generate_array_data.py — 2 jammers, realistic geometry, 2x2 URA
- music_spectrum.py — fixed with normalisation + diagonal loading
- mvdr_beamformer.py — done
- hybrid_sim.py — done  
- run_all.py — done

GITHUB: git@github.com:itsstoomuch/antiJAMsimulationV2.git