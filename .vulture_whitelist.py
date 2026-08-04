"""
Vulture whitelist file.

Add entries here for code that vulture incorrectly identifies as unused.
Format: name  # noqa - comment explaining why it is used

Everything below is an AnomalyMatch config attribute. The scripts here only
*write* them onto the `DotMap` config; they are read inside AnomalyMatch (or in
the training / prediction subprocesses it spawns), which vulture cannot see.
"""

# Session and data-source selection
_.log_level  # noqa - read by Session.__init__ -> set_log_level
_.training_data_source  # noqa - read by create_training_data_source
_.test_ratio  # noqa - read by training_process.py to decide on a test split
_.save_dir  # noqa - read by SessionIOHandler when resolving session paths

# Normalisation, consumed by fitsbolt via get_fitsbolt_config / validate_config
_.channel_combination  # noqa - combines the 4 Cutana bands into 3 channels
_.num_channels  # noqa - read by the training subprocess when building the net
_.fits_extension  # noqa - read by fitsbolt when loading FITS extensions
_.cutout_padding_factor  # noqa - read by Cutana when extracting cutouts
