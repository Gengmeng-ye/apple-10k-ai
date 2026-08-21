"""Run Apple's financial data pipeline."""

from src.transform_financials import main as transform_financials
from src.validate_data import main as validate_data


def main() -> None:
    """Run transformation and validation in order."""
    print("\nStep 1: Transforming financial data...")
    transform_financials()

    print("\nStep 2: Validating financial data...")
    validate_data()

    print("\nFinancial data pipeline completed successfully.")


if __name__ == "__main__":
    main()