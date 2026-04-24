import pandas as pd
import matchingtools as mt

# Replace with your actual data path or existing DataFrame.
# df = mt.load_data("lab04-DATAlab_atualizado.dta")

# Example after loading df:
# covariates = ["age", "gender", "hdi", "region", "income"]
# m = mt.matchit(
#     df,
#     treatment="bf",
#     covariates=covariates,
#     formula="bf ~ age + C(gender) + hdi + C(region) + C(income) + I(hdi ** 2)",
#     method="nearest",
#     ratio=1,
#     replace=True,
# )
# dm = mt.match_data(m)
# print(m.summary())
# print(mt.balance_table(m))
# fig, ax = mt.love_plot(m)
# print(mt.estimate_att(dm, outcome="lula", treatment="bf").summary())
