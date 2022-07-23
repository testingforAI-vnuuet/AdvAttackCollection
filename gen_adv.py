import argparse

from src.adv_generator import AdvGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Config file path for training autoencoder-based reformer')

    parser.add_argument('config_filepath', type=str,
                        help='A required string file path')

    args = parser.parse_args()

    adv_generator = AdvGenerator(args.config_filepath,
                                 limit_origins=100)
    adv_generator.attack()
