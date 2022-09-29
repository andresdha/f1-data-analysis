from src.telemetry_comparer import QualifyingComparer

# Choose Drivers to compare
drivers = ["LEC", "VER"]

# Choose a Grand Prix
GRAND_PRIX = "Monza"

# Choose a YEAR for the GP
YEAR = 2022

monza_2022_comparer = QualifyingComparer(drivers, GRAND_PRIX, YEAR)

monza_2022_comparer.process_telemetry()
monza_2022_comparer.process_minisectors()
monza_2022_comparer.compare_telemetry()
monza_2022_comparer.compare_minisectors()
