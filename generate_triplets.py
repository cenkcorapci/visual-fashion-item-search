import argparse

import pandas as pd

from commons.config import MVC_GENERATED_EASY_TRIPLETS_CSV, MVC_GENERATED_HARD_TRIPLETS_CSV
from data.triplet_generator import generate_triplets

# Get Parameters ---------------------------------------------
usage_docs = """
--easy <integer> 1 if yes 0 if no
"""

parser = argparse.ArgumentParser(usage=usage_docs)

parser.add_argument('--easy', type=int, default=1)

args = parser.parse_args()
data_set = generate_triplets(easy=args.easy == 1)
path = MVC_GENERATED_EASY_TRIPLETS_CSV if args.easy == 1 else MVC_GENERATED_HARD_TRIPLETS_CSV
pd.DataFrame(data_set, columns=['anchor', 'positive', 'negative']).to_csv(MVC_GENERATED_EASY_TRIPLETS_CSV, index=False)
