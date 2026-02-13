from fandom_span_id_retrieval.utils.seed_utils import set_seed


def test_set_seed_runs() -> None:
    set_seed(123, deterministic=True)
