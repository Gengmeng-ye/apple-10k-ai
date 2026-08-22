"""Run Apple's financial and risk analysis pipeline."""

from src.extract_risk_factors import (
    main as extract_risk_factors,
)
from src.model_risk_topics import (
    main as model_risk_topics,
)
from src.transform_financials import (
    main as transform_financials,
)
from src.validate_data import (
    main as validate_data,
)


def main() -> None:
    """Run financial and risk analysis steps in order."""
    print("\nStep 1: Transforming financial data...")
    transform_financials()

    print("\nStep 2: Validating financial data...")
    validate_data()

    print("\nStep 3: Extracting Risk Factors...")
    extract_risk_factors()

    print("\nStep 4: Modeling risk topics...")
    model_risk_topics()

    print(
        "\nFinancial and risk analysis pipeline "
        "completed successfully."
    )


if __name__ == "__main__":
    main()