import learning_envs
import sys

if __name__ == "__main__":
    # Show user's passed arguments
    args = sys.argv
    print(f"\npassed args: {args[1:]}\n\n")

    # Select AI learning environment
    print("Load environment:\n")
    print(f"{'ID':<10}{'Name':<15}Path")

    for id,(name,info) in enumerate(learning_envs.environments.items()):
        print(f"{id:<10}{name:<15}{info['path']}")

    to_load = input("\nSelect environment: ")

    # Input checking
    try:
        to_load = int(to_load)
    except:
        exit("\nUNABLE TO LOAD ENVIRONMENT: Input MUST be integer.")

    if to_load >= len(learning_envs.environments) or to_load < 0:
        exit(f"\nUNABLE TO LOAD ENVIRONMENT: Input must be from 0-{len(learning_envs.environments)}")

    # Select environment
    selected_env = list(learning_envs.environments)[to_load]
    module = learning_envs.environments[selected_env]['module']

    # Environment checker
    if not hasattr(module, "main"):
        exit("\nUNABLE TO LOAD ENVIRONMENT: Environment not setup correctly. Missing 'main.py' in __init__.py")

    # Run environment
    module.main(args)