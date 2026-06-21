TRAIN_DATA = [
    (
        "ABC Technologies Pvt Ltd signed an agreement with XYZ Retail Pvt Ltd.",
        {
            "entities": [
                (0, 24, "PARTY"),
                (51, 69, "PARTY")
            ]
        }
    ),
    (
        "The contract started on January 15, 2026.",
        {
            "entities": [
                (24, 40, "DATE")
            ]
        }
    ),
    (
        "The client paid Rs. 5,00,000 for the project.",
        {
            "entities": [
                (17, 30, "MONEY")
            ]
        }
    ),
    (
        "Termination requires a notice period of 30 days.",
        {
            "entities": [
                (40, 47, "DURATION")
            ]
        }
    )
]