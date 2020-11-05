from implicit.als import AlternatingLeastSquares
from pathlib import Path
from scipy import stats
import scipy.sparse as sparse
import pandas as pd
import numpy as np

# Matrix factorization using Alternating Least Squares(ALS) with implicit.als

def build_product_user_matrix(df_user_product_prior):
    """
    Generates a utility matrix with prior data set.
    Rows: products & Column: users respectively.
    """
    print("Creating product user matrix ...")

    product_user_matrix = sparse.coo_matrix((df_user_product_prior["quantity"],
                                            (df_user_product_prior["product_id"].cat.codes.copy(),
                                             df_user_product_prior["user_id"].cat.codes.copy())))
    sparse.save_npz(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/product_user_matrix.npz",
        product_user_matrix)

def sparsity(matrix):
    """
    Caluculate the sparsity of the given matrix
    """
    total_size = matrix.shape[0] * matrix.shape[1]
    actual_size = matrix.size
    sparsity = (1 - (actual_size / total_size)) * 100
    return sparsity

def confidence_matrix(prod_user_matrix, alpha):
    """
    Returns a confidence matrix with the given a utility matrix
    c_ui = 1 + α * r_ui
    """
    return (prod_user_matrix * alpha).astype("double")


def build_imf(prod_user_matrix, alpha):
    """
    Builds models with the utility matrix and model parameters
    """
    # Build model
    print("Building IMF model with alpha: {} ...".format(alpha))
    model = AlternatingLeastSquares(factors=75, regularization=0.01, iterations=15,use_cg=True)
    model.approximate_similar_items = False
    model.fit(confidence_matrix(prod_user_matrix, alpha))
    return model

def new_products(row):
    """
    Given a row of the validation dataset
    Returns the list of new products purchased
    """
    actual = row["products"][1:-1]  # Products purchased currently
    actual = set([int(p.strip()) for p in actual.strip().split(",")])
    liked = set([p_dict[i] for i in user_product_matrix[u_dict[row["user_id"]]].indices])  # User's purchase history
    return actual - liked  # Return only new products purchased

def get_score(actual, pred):
    """
    Caluculate the f1 score of prediction
    """
    if len(actual) == 0:
        return 0
    actual, pred = set(actual), set(pred)
    recall = len(actual.intersection(pred)) / len(actual)
    precision = len(actual.intersection(pred)) / len(pred)
    if precision+recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    else:
        return 0

def imf_recommend(row):
    """
    Return the f1 score with a given row in the vslidation dataset (a single user ID)
    """
    actual = new_products(row)
    recommended = imf_model.recommend(u_dict[row["user_id"]], user_product_matrix, N=10)
    recommended = [p_dict[r[0]] for r in recommended]
    return get_score(actual, recommended)

def popular_recommend(row):
    """
    Returns f1 score of baseline recommendation (to a single user ID)
    """
    actual = new_products(row)
    return get_score(actual, popular_products)

def build_eval_df(user_product_validation,n1,n2):
    print("Making prediction on validation data ...")
    df_eval = user_product_validation[n1:n2].copy()
    df_eval["popular_score"] = df_eval.apply(popular_recommend, axis=1)
    df_eval["imf_f1_score"] = df_eval.apply(imf_recommend, axis=1)
    return df_eval

if __name__ == "__main__":
    user_product_prior_raw = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/user_product_prior_raw.csv")
    user_product_prior = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/user_product_prior.csv")
    user_product_prior["user_id"] = user_product_prior["user_id"].astype("category")
    user_product_prior["product_id"] = user_product_prior["product_id"].astype("category")
    products = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/products.csv")
    user_product_validation = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/user_products_validation.csv")
    order_products_prior = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/order_products__prior.csv")
    #order_products_prior = pd.merge(order_products_prior, products, on="product_id", how="left")
    #order_products_prior.to_csv(
        #"/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/order_products_prior.csv"
       # , index_label=False)
    order_products_prior =  pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/order_products_prior.csv")

    # build product user matrix & user product matrix
    product_user_matrix = sparse.load_npz(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/product_user_matrix.npz").tocsr()
    user_product_matrix = product_user_matrix.T.tocsr()
    p_u_matrix_sparsity = sparsity(product_user_matrix)
    print("Sparsity of product-user matrix is",p_u_matrix_sparsity)

    # build model
    alpha = 30
    imf_model = build_imf(product_user_matrix, alpha)

    # Maps user_id: user index
    u_dict = {uid: i for i, uid in enumerate(user_product_prior["user_id"].cat.categories)}
    # Maps product_index: product id
    p_dict = dict(enumerate(user_product_prior["product_id"].cat.categories))

    # Baseline: recommend the top 10 products
    popular_products = list(order_products_prior["product_id"].value_counts().head(10).index)

    # prediction with validation set & mean f1 scores
    validation_score1 = build_eval_df(user_product_validation,1,20000)
    validation_mean_f1_1 = np.mean(validation_score1["imf_f1_score"])
    baseline_mean_f1_1 = np.mean(validation_score1["popular_score"])
    print("imf f1 score(1:20000): {:.2f}%".format(validation_mean_f1_1 * 100))
    print("Baseline f1 score(1:20000): {:.2f}%".format(baseline_mean_f1_1 * 100))
    validation_score2 = build_eval_df(user_product_validation,20001,40000)
    validation_mean_f1_2 = np.mean(validation_score2["imf_f1_score"])
    baseline_mean_f1_2 = np.mean(validation_score2["popular_score"])
    print("imf f1 score(20001:40000): {:.2f}%".format(validation_mean_f1_2 * 100))
    print("baseline f1 score(20001:40000): {:.2f}%".format(baseline_mean_f1_2 * 100))

    '''
        # prediction sample: user_id = 1
        user_id = 1
        recommendations = imf_model.recommend(u_dict[user_id], user_product_matrix, N=10)
        # Actual
        row = user_product_validation.loc[user_product_validation.user_id == user_id]
        actual = list(row["products"])
        actual = actual[0][1:-1]
        actual = list(np.array([p.strip() for p in actual.strip().split(",")]).astype(np.int64))
        act_products = []
        for pid in actual:
            act_products.extend((products.loc[products.product_id == pid].product_name).tolist())
        print("Actual products bought by user {}\n{}".format(user_id, act_products))

        # Recommended
        r = [p_dict[r[0]] for r in recommendations]  # Takes the product_cat_code and maps to product_id
        rec_products = []
        for pid in r:
            rec_products.extend((products.loc[products.product_id == pid].product_name).tolist())
        print("\nRecommendations for user {}\n{}".format(user_id, rec_products))
        f1_example = get_score(act_products, rec_products)
        print("\nF1_score for user {}\n{}".format(user_id, f1_example*100))
    '''