from .SDF import SDF, SDF_CM, SDF_Hysteresis, SDF_Trend, SDF_TSA, SDF_Window

solutions = {
    "SDF": SDF.main,
    "SDF_CM": SDF_CM.main,
    "SDF_Hysteresis": SDF_Hysteresis.main,
    "SDF_Trend": SDF_Trend.main,
    "SDF_TSA": SDF_TSA.main,
    "SDF_Window": SDF_Window.main,
}

for idx, sol in enumerate(solutions):
    print(f"  - ({idx}) {sol}")

idx = int(input("Choose solution: "))

solution = list(solutions.keys())[idx]

main = solutions[solution]

print(f"Running solution {solution}...")

main()

