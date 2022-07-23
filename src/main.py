from src.adv_generator import AdvGenerator

if __name__ == "__main__":
    adv_generator = AdvGenerator("D:\Things\PyProject\AdvAttackCollection\config.ini",
                                 limit_origins=100)
    adv_generator.attack()
