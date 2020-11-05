import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import cart
import itertools
import time
from random import randint


class TopNRecommendation(object):
    def __init__(self, product_factors, user_factors, product_user_matrix):
        self.product_factors = product_factors
        self.user_factors = user_factors
        self.product_user_matrix = product_user_matrix

    def recommend(self, user_id, N=10):
        """
        Finds top N Recommendations
        """
        scores = self.user_factors[user_id] @ self.product_factors.T
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])

    def recommend_new(self, user_id, N=10):
        """
        Finds Top N new Recommendations
        """
        scores = self.user_factors[user_id] @ self.product_factors.T
        bought_indices = self.product_user_matrix.T[user_id].nonzero()[1]
        count = N + len(bought_indices)
        ids = np.argpartition(scores, -count)[-count:]
        best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in bought_indices), N))


def map_matrix_id(df_user_product_prior):
    u_dict = {uid: i for i, uid in enumerate(df_user_product_prior["user_id"].cat.categories)}
    p_dict = dict(enumerate(df_user_product_prior["product_id"].cat.categories))
    return u_dict, p_dict


def actual_products(df_user_product_valid, df_products, user_id):
    # Actual
    row = df_user_product_valid.loc[df_user_product_valid.user_id == user_id]
    actual = list(row["products"])
    actual = actual[0][1:-1]
    actual = list(np.array([p.strip() for p in actual.strip().split(",")]).astype(np.int64))
    act_products = []
    for pid in actual:
        act_products.extend(df_products.loc[df_products.product_id == pid].product_name.tolist())
    print("Actual products bought by user {}\n{}\n\n".format(user_id, act_products))


def rec_products(recommendations, df_products, p_dict, user_id):
    # All Products Recommended
    rec_products = []
    for rec in recommendations:
        print(rec)
        rec_products.extend(df_products.loc[p_dict[rec[0]] == df_products.product_id].product_name.tolist())
    print("All products recommended to user {}\n{}\n\n".format(user_id, rec_products))


def get_k_popular(k, df_order_product_prior):
    popular_products = list(df_order_product_prior["product_id"].value_counts().head(k).index)
    return popular_products


def print_goods(popular_products):
    print('10 most popular products on the platform is,')
    popular_goods = []
    print(popular_products)
    for rec in popular_products:
        print(rec)
        popular_goods.extend(df_products.loc[p_dict[rec]+1 == df_products.product_id].product_name.tolist())
    print(popular_goods)


def recall(bought, pred):
    if len(bought) == 0:
        return 0
    bought, pred = set(bought), set(pred)
    return len(bought.intersection(pred)) / len(bought)


def precision(bought, pred):
    if len(pred) == 0:
        return 0
    bought, pred = set(bought), set(pred)
    return len(bought.intersection(pred))/len(pred)


def f1(bought, pred):
    a = precision(bought, pred)
    b = recall(bought, pred)
    if a+b == 0:
        return 0
    else:
        return 2 * (a * b)/(a + b)


def new_purchase_row(row):
    """
    Given a row in the validation set
    Returns the list of new products purchased
    """
    actual = row["products"][1:-1]  # Products purchased currently
    actual = set([int(p.strip()) for p in actual.strip().split(",")])
    liked = set([p_dict[i] for i in user_product_matrix[u_dict[row["user_id"]]].indices])  # User's purchase history
    new_purchase = actual - liked
    return new_purchase


def popular_recommend(row):
    """
    Given a row in the test dataset
    Returns the recall score when popular products are recommended
    """
    actual = new_purchase_row(row)
    return f1(actual, popular_products)


def svd_recommend_new(row):
    """
    Given a row in the test dataset
    Returns the recall score when our model recommends new products
    """
    actual = new_purchase_row(row)
    recommended = svd_rec.recommend_new(u_dict[row["user_id"]], N=10)
    recommended = [p_dict[r[0]] for r in recommended]
    return f1(actual, recommended)


def build_eval_df(user_product_validation, n1, n2):
    start = time.time()
    print("Making prediction on validation data ...")
    df_eval = user_product_validation[n1:n2].copy()
    df_eval["popular_score"] = df_eval.apply(popular_recommend, axis=1)
    df_eval["svd_new_score"] = df_eval.apply(svd_recommend_new, axis=1)
    print("Completed in {:.2f}s".format(time.time() - start))
    return df_eval


if __name__ == "__main__":
    # Order datasets
    df_order_products_prior = pd.read_csv("order_products__prior.csv")
    df_order_products_train = pd.read_csv("order_products__train.csv")
    df_orders = pd.read_csv("orders.csv")

    # Products
    df_products = pd.read_csv("products.csv")

    # user_product sets
    df_user_product_valid = pd.read_csv("user_product_valid.csv").astype('category')
    df_user_product_train = pd.read_csv("user_product_prior.csv").astype('category')
    product_user_matrix = sparse.load_npz("product_user_matrix.npz").tocsr().astype(np.float32)
    user_product_matrix = product_user_matrix.T.tocsr()
    u_dict, p_dict = map_matrix_id(df_user_product_train)

    # print out the 10 most popular products on the platform
    top_k = 10
    popular_products = get_k_popular(top_k, df_order_products_prior)
    print_goods(popular_products)

    # tune factor number
    lambdas = [3, 5, 10, 25]
    validation_mean = []
    baseline_mean = []
    for lam in lambdas:
        product_factors, sigma, user_factors = linalg.svds(product_user_matrix, lam)
        user_factors = user_factors.T * sigma
        svd_rec = TopNRecommendation(product_factors, user_factors, product_user_matrix)
        v_mean_per_lam = []
        b_mean_per_lam = []
        for i in range(1):
            k = randint(1, 2000)
            validation_score1 = build_eval_df(df_user_product_valid, 20001, 40000)
            validation_mean_svd_1 = np.mean(validation_score1["svd_new_score"])
            baseline_mean_1 = np.mean(validation_score1["popular_score"])
            v_mean_per_lam.append(validation_mean_svd_1)
            b_mean_per_lam.append(baseline_mean_1)
        v_mean = sum(v_mean_per_lam)/len(v_mean_per_lam)
        b_mean = sum(b_mean_per_lam)/len(b_mean_per_lam)
        validation_mean.append(v_mean)
        baseline_mean.append(b_mean)
    print("svd f1 score", validation_mean)
    print("Baseline f1 score", baseline_mean)

