"""Run a Python LensKit analogue of the recommender-evaluation assignment."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from lenskit.basic import BiasConfig, BiasScorer, PopConfig, PopScorer
from lenskit.batch import predict, recommend
from lenskit.data import ItemList, RecQuery, from_interactions_df
from lenskit.knn import ItemKNNConfig, ItemKNNScorer, UserKNNConfig, UserKNNScorer
from lenskit.pipeline import topn_pipeline


NEIGHBORHOODS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
LIST_SIZE = 10


def dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def user_metrics(
    predicted: dict[int, float],
    recommended: list[int],
    truth: dict[int, float],
    tag_counts: dict[int, dict[str, int]],
) -> dict[str, float]:
    valid_predictions = [(item, score) for item, score in predicted.items() if math.isfinite(score)]
    if valid_predictions:
        errors = [(score - truth[item]) ** 2 for item, score in valid_predictions if item in truth]
        rmse = math.sqrt(sum(errors) / len(errors)) if errors else math.nan
        ranked_test = sorted(
            ((item, score) for item, score in valid_predictions if item in truth),
            key=lambda pair: pair[1],
            reverse=True,
        )
        pred_gains = [truth[item] for item, _ in ranked_test]
        ideal_gains = sorted(truth.values(), reverse=True)
        pred_ndcg = dcg(pred_gains) / dcg(ideal_gains) if dcg(ideal_gains) else 0.0
    else:
        rmse = math.nan
        pred_ndcg = math.nan

    top_items = recommended[:LIST_SIZE]
    top_gains = [truth.get(item, 0.0) for item in top_items]
    ideal_gains = sorted(truth.values(), reverse=True)[:LIST_SIZE]
    top_ndcg = dcg(top_gains) / dcg(ideal_gains) if dcg(ideal_gains) else 0.0
    relevant = set(truth)
    hits = 0
    precision_sum = 0.0
    reciprocal_rank = 0.0
    for rank, item in enumerate(top_items, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank
    avg_precision = precision_sum / min(len(relevant), LIST_SIZE) if relevant else 0.0

    tag_probabilities: dict[str, float] = {}
    if top_items:
        for item in top_items:
            counts = tag_counts.get(item, {})
            total = sum(counts.values())
            if total:
                for tag, count in counts.items():
                    tag_probabilities[tag] = tag_probabilities.get(tag, 0.0) + (
                        count / total / len(top_items)
                    )
    entropy = -sum(prob * math.log2(prob) for prob in tag_probabilities.values() if prob > 0)

    return {
        "RMSE.ByUser": rmse,
        "Predict.nDCG": pred_ndcg,
        "TopN.nDCG": top_ndcg,
        "MRR": reciprocal_rank,
        "MAP": avg_precision,
        "TagEntropy": entropy,
    }


def make_holdout_folds(
    ratings: pd.DataFrame, folds: int, seed: int, holdout_per_user: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    result = []
    for fold in range(folds):
        rng = np.random.default_rng(seed + fold)
        held_out: list[int] = []
        for _, user_rows in ratings.groupby("user_id", sort=False):
            indexes = user_rows.index.to_numpy()
            count = min(holdout_per_user, max(0, len(indexes) - 1))
            if count:
                held_out.extend(rng.choice(indexes, size=count, replace=False).tolist())
        test = ratings.loc[held_out].copy()
        train = ratings.drop(index=held_out).copy()
        result.append((train, test))
    return result


def build_tag_similarity(
    items: list[int], tags: pd.DataFrame
) -> tuple[csr_matrix, dict[int, int], dict[int, dict[str, int]]]:
    item_index = {item: index for index, item in enumerate(items)}
    tag_counts: dict[int, dict[str, int]] = {}
    for row in tags.itertuples(index=False):
        item = int(row.movieId)
        tag = str(row.tag).lower()
        per_item = tag_counts.setdefault(item, {})
        per_item[tag] = per_item.get(tag, 0) + 1

    vocabulary = sorted({tag for counts in tag_counts.values() for tag in counts})
    tag_index = {tag: index for index, tag in enumerate(vocabulary)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    document_frequency = np.zeros(len(vocabulary), dtype=np.float64)
    for item, counts in tag_counts.items():
        if item not in item_index:
            continue
        for tag, count in counts.items():
            rows.append(item_index[item])
            cols.append(tag_index[tag])
            values.append(float(count))
            document_frequency[tag_index[tag]] += 1

    matrix = coo_matrix(
        (values, (rows, cols)), shape=(len(items), len(vocabulary)), dtype=np.float64
    ).tocsr()
    if vocabulary:
        idf = np.log((1 + len(items)) / (1 + document_frequency)) + 1.0
        matrix = matrix.multiply(idf).tocsr()
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        matrix = matrix.multiply(1.0 / norms[:, None]).tocsr()
    similarities = (matrix @ matrix.T).tocsr()
    return similarities, item_index, tag_counts


def content_scores_by_configuration(
    candidate_items: list[int],
    user_history: pd.DataFrame,
    item_means: dict[int, float],
    global_mean: float,
    similarities: csr_matrix,
    item_index: dict[int, int],
    neighborhoods: list[int],
) -> dict[tuple[bool, int], dict[int, float]]:
    history_items = [int(item) for item in user_history["item_id"] if int(item) in item_index]
    history_ratings = user_history.set_index("item_id")["rating"]
    known_items = [item for item in history_items if item in history_ratings.index]
    configurations = {
        (normalize, neighbors): {}
        for normalize in (False, True)
        for neighbors in neighborhoods
    }
    if not known_items:
        fallback = {item: item_means.get(item, global_mean) for item in candidate_items}
        return {key: fallback.copy() for key in configurations}

    history_rows = [item_index[item] for item in known_items]
    raw_ratings = np.array([float(history_ratings.loc[item]) for item in known_items])
    candidate_rows = [item_index[item] for item in candidate_items]
    weights = similarities[candidate_rows, :][:, history_rows].toarray()
    residual_ratings = raw_ratings - np.array(
        [item_means.get(item, global_mean) for item in known_items]
    )
    for row, item in enumerate(candidate_items):
        positive = np.flatnonzero(weights[row] > 0)
        fallback = item_means.get(item, global_mean)
        if not len(positive):
            for scores in configurations.values():
                scores[item] = fallback
            continue
        order = positive[np.argsort(weights[row, positive])[::-1]]
        ordered_weights = weights[row, order]
        denominators = np.cumsum(ordered_weights)
        raw_numerators = np.cumsum(ordered_weights * raw_ratings[order])
        residual_numerators = np.cumsum(ordered_weights * residual_ratings[order])
        for (normalize, neighbors), scores in configurations.items():
            index = min(neighbors, len(order)) - 1
            value = (
                residual_numerators[index] / denominators[index] + fallback
                if normalize
                else raw_numerators[index] / denominators[index]
            )
            scores[item] = float(value)
    return configurations


def evaluate_fold(
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    catalog: list[int],
    all_users: list[int],
    tag_counts: dict[int, dict[str, int]],
    similarities: csr_matrix,
    item_index: dict[int, int],
    neighborhoods: list[int],
) -> list[dict[str, object]]:
    dataset = from_interactions_df(train, users=all_users, items=catalog)
    train_by_user = {user: rows for user, rows in train.groupby("user_id", sort=False)}
    test_by_user = {user: rows for user, rows in test.groupby("user_id", sort=False)}
    users = [int(user) for user in test_by_user]
    test_pairs = test[["user_id", "item_id"]].copy()
    queries: list[tuple[RecQuery, ItemList]] = []
    candidate_by_user: dict[int, list[int]] = {}
    for user in users:
        seen = set(int(item) for item in train_by_user.get(user, pd.DataFrame()).get("item_id", []))
        candidates = [item for item in catalog if item not in seen]
        candidate_by_user[user] = candidates
        queries.append((RecQuery(user_id=user), ItemList(candidates)))

    global_mean = float(train["rating"].mean())
    item_means = train.groupby("item_id")["rating"].mean().to_dict()
    rows: list[dict[str, object]] = []

    model_specs = [
        ("GlobalMean", BiasScorer(BiasConfig(entities=set())), True, False),
        ("ItemMean", BiasScorer(BiasConfig(entities={"item"})), True, True),
        ("PersMean", BiasScorer(BiasConfig(entities={"user", "item"})), True, True),
        ("Popular", PopScorer(PopConfig(score="count")), False, True),
    ]
    user_scorer = UserKNNScorer(UserKNNConfig(max_nbrs=max(neighborhoods)))
    item_scorer = ItemKNNScorer(ItemKNNConfig(max_nbrs=max(neighborhoods)))

    for name, scorer, do_predict, do_recommend in model_specs:
        pipeline = topn_pipeline(
            scorer, predicts_ratings=do_predict, n=LIST_SIZE, name=name
        )
        pipeline.train(dataset)
        pred = predict(pipeline, test_pairs, n_jobs=1).to_df() if do_predict else pd.DataFrame()
        recs = recommend(pipeline, queries, n=LIST_SIZE, n_jobs=1).to_df() if do_recommend else pd.DataFrame()
        rows.append(
            summarize_configuration(
                fold, name, None, test_by_user, users, pred, recs, tag_counts
            )
        )

    for name, scorer in [("UserUserCosine", user_scorer), ("ItemItem", item_scorer)]:
        pipeline = topn_pipeline(
            scorer, predicts_ratings=True, n=LIST_SIZE, name=name
        )
        pipeline.train(dataset)
        for neighbors in neighborhoods:
            scorer.config.max_nbrs = neighbors
            pred = predict(pipeline, test_pairs, n_jobs=1).to_df()
            recs = recommend(pipeline, queries, n=LIST_SIZE, n_jobs=1).to_df()
            rows.append(
                summarize_configuration(
                    fold, name, neighbors, test_by_user, users, pred, recs, tag_counts
                )
            )

    content_results = {
        (normalized, neighbors): {"predictions": [], "recommendations": []}
        for normalized in (False, True)
        for neighbors in neighborhoods
    }
    for user in users:
        history = train_by_user.get(user, pd.DataFrame(columns=train.columns))
        scores_by_configuration = content_scores_by_configuration(
            candidate_by_user[user],
            history,
            item_means,
            global_mean,
            similarities,
            item_index,
            neighborhoods,
        )
        truth_items = set(int(item) for item in test_by_user[user]["item_id"])
        for (normalized, neighbors), scores in scores_by_configuration.items():
            result = content_results[(normalized, neighbors)]
            result["predictions"].extend(
                {"user_id": user, "item_id": item, "score": scores[item]}
                for item in truth_items
                if item in scores
            )
            ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
            result["recommendations"].extend(
                {"user_id": user, "item_id": item, "rank": rank}
                for rank, (item, _) in enumerate(ranked[:LIST_SIZE], start=1)
            )
    for (normalized, neighbors), result in content_results.items():
        name = "TagContentNorm" if normalized else "TagContent"
        rows.append(
            summarize_configuration(
                fold,
                name,
                neighbors,
                test_by_user,
                users,
                pd.DataFrame(result["predictions"]),
                pd.DataFrame(result["recommendations"]),
                tag_counts,
            )
        )

    return rows


def summarize_configuration(
    fold: int,
    name: str,
    neighbors: int | None,
    test_by_user: dict[int, pd.DataFrame],
    users: list[int],
    predictions: pd.DataFrame,
    recommendations: pd.DataFrame,
    tag_counts: dict[int, dict[str, int]],
) -> dict[str, object]:
    prediction_map: dict[int, dict[int, float]] = {}
    if not predictions.empty and {"user_id", "item_id", "score"} <= set(predictions.columns):
        for row in predictions.itertuples(index=False):
            prediction_map.setdefault(int(row.user_id), {})[int(row.item_id)] = float(row.score)
    recommendation_map: dict[int, list[int]] = {}
    if not recommendations.empty and {"user_id", "item_id"} <= set(recommendations.columns):
        ordered = recommendations.sort_values(
            ["user_id", "rank"] if "rank" in recommendations.columns else ["user_id", "score"],
            ascending=[True, True] if "rank" in recommendations.columns else [True, False],
        )
        for user, user_rows in ordered.groupby("user_id", sort=False):
            recommendation_map[int(user)] = [int(item) for item in user_rows["item_id"]]

    per_user: list[dict[str, float]] = []
    prediction_count = 0
    expected_predictions = 0
    for user in users:
        truth = {
            int(row.item_id): float(row.rating)
            for row in test_by_user[user].itertuples(index=False)
        }
        predicted = prediction_map.get(user, {})
        prediction_count += sum(math.isfinite(score) for score in predicted.values())
        expected_predictions += len(truth)
        per_user.append(
            user_metrics(
                predicted,
                recommendation_map.get(user, []),
                truth,
                tag_counts,
            )
        )

    metric_names = ["RMSE.ByUser", "Predict.nDCG", "TopN.nDCG", "MRR", "MAP", "TagEntropy"]
    means = {
        metric: float(np.nanmean([values[metric] for values in per_user]))
        if any(math.isfinite(values[metric]) for values in per_user)
        else math.nan
        for metric in metric_names
    }
    if name == "GlobalMean":
        means.update(
            {
                "TopN.nDCG": math.nan,
                "MRR": math.nan,
                "MAP": math.nan,
                "TagEntropy": math.nan,
            }
        )
    return {
        "Partition": fold,
        "Algorithm": name,
        "NNbrs": neighbors,
        "Coverage": prediction_count / expected_predictions if expected_predictions else math.nan,
        **means,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--holdout-per-user", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--neighborhoods", nargs="*", type=int, default=NEIGHBORHOODS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/eval-results.csv"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ratings = pd.read_csv(root / "data" / "ratings.csv").rename(
        columns={"userId": "user_id", "movieId": "item_id"}
    )
    tags = pd.read_csv(root / "data" / "tags.csv", encoding="latin1")
    movies = pd.read_csv(
        root / "data" / "movies.csv", usecols=["movieId"], encoding="latin1"
    )
    ratings = ratings[["user_id", "item_id", "rating"]].dropna()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(int)
    tags["movieId"] = tags["movieId"].astype(int)
    catalog = sorted(int(item) for item in movies["movieId"].unique())
    all_users = sorted(int(user) for user in ratings["user_id"].unique())
    similarities, item_index, tag_counts = build_tag_similarity(catalog, tags)

    result_rows: list[dict[str, object]] = []
    folds = make_holdout_folds(ratings, args.folds, args.seed, args.holdout_per_user)
    for fold, (train, test) in enumerate(folds, start=1):
        print(
            f"Fold {fold}/{len(folds)}: {len(train):,} train ratings, "
            f"{len(test):,} test ratings",
            flush=True,
        )
        result_rows.extend(
            evaluate_fold(
                fold,
                train,
                test,
                catalog,
                all_users,
                tag_counts,
                similarities,
                item_index,
                args.neighborhoods,
            )
        )

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(output, index=False)
    print(f"Wrote {len(result_rows)} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
