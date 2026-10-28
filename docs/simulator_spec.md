# Simulator specification

## Goal
Build a semi-synthetic simulator for copy-trading recommendation experiments.

## Main entities
- User
- Strategy
- Exposure
- Copy outcome

## Simulation pipeline
1. Generate strategies
2. Generate users
3. Compute compatibility
4. Generate exposures
5. Simulate copy decisions
6. Simulate outcomes
7. Export tables
8. Validate generated data

## Output tables
- users.csv
- strategies.csv
- exposures.csv
- copy_outcomes.csv
- metadata.json