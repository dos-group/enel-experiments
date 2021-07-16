# enel-experiments

Prototypical implementation of "Enel" for runtime prediction / resource allocation / dynamic scaling. Please definitely consider reaching out if you have questions or encounter problems. We are happy to help anytime!

This repository contains several subdirectories, featuring the following content:

- `data`: The data we recorded during our experiments, or, to be more precise, needed for our evalation.
- `enel_injector`: A small java program handling the injection of failures.
- `enel_service`: Our python web service that handles training of models + submission & adjustments of spark applications.
- `evaluation`: Python notebooks for the evaluation.
- `spark_utils`: A package that encompasses benchmark jobs, dataset generators, and custom spark listeners that we have used.

Except for `data` and `evaluation`, all subdirectories contain further information. 