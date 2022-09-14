import learning_envs.dot.game as game
import learning_envs.dot.visual as visual
import learning_envs.dot.showoff as showoff
import learning_envs.dot.processing as processing

info = {
    "load [!generation]" : f"Load a previously saved generationt to continue from there",
    "showoff [?seed]"    : f"Make all saved best networks compete with eachother"
}

def main(argv:list):
    print("DOT help:")
    for name,desc in info.items():
        print(f"{name:<20}{desc}")

    if argv[1:] and argv[1] == "showoff":
        showoff.main(argv)

    elif argv[1:] and argv[1] == "processing":
        processing.main(argv)
    else:
        visual.main(argv)
    