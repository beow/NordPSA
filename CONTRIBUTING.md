# Contributing to NordPSA

Thanks for your interest. NordPSA is maintained by one person, so contributions are reviewed as time allows.

## Questions and bug reports

Open an [issue](https://github.com/beow/NordPSA/issues) first — for bugs, questionable assumptions or ideas.
Please include the command you ran, the resolved settings (`results/<run>/run_config.yaml` or
`nordpsa expand --print-config`) and the relevant part of `run.log`.

For larger changes, open an issue before writing code so we can agree on the approach.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep it focused: one topic per pull request.
3. Run the unit tests: `pytest` (no input data needed). They also run on every pull request.
4. Open a pull request describing **what** changed and **why**.

Conventions:

- Commit messages and code comments in English.
- New settings get their default in `config/defaults.yaml` (exactly one place); named deviations go in
  `config/experiments/`. Do not change model data in `config/zones.yaml` without a source.
- If the change affects model results, say how: which run(s) you compared and what moved (capacities, prices,
  system cost). Results themselves (`results/`) are not committed.
- Input data is not in the repository; `make data` fetches it (some sources need API tokens, see the `Makefile`).

## License

By contributing you agree that your contribution is licensed under the MIT License (see `LICENSE`).
