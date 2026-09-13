"""The write side of the malatium data store.

One module per table, each with a `run(...)` taking at most a start and an
end date, wired into one click CLI:

    uv run pipelines --help
    uv run pipelines option-greeks
    uv run pipelines build

Every step writes a bear-lake table under `ML_DATA_STORE` and reads only
other tables in the same store. `store.py` is the one place the table
definitions and the on-disk layout are spelled.
"""
