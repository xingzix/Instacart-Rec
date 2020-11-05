import numpy as np
import csv
import util
import pandas as pd
import scipy.sparse as sparse


def make_valid_set(savepath, df_orders, df_order_product_train):
    # select eval set == train
    df_order_user_curr = df_orders.loc[df_orders.eval_set == "train"].reset_index()
    # only select order_id and user_id
    df_order_user_curr = df_order_user_curr[["order_id", "user_id"]]
    df_order_product_valid = df_order_product_train[["order_id", "product_id"]]
    df_order_product_valid = df_order_product_valid.groupby("order_id")["product_id"].apply(list).reset_index().rename(
        columns={"product_id": "products"})

    # Merge on order id
    df_user_product_valid = pd.merge(df_order_user_curr, df_order_product_valid, on="order_id")
    df_user_product_valid = df_user_product_valid[["user_id", "products"]]

    df_user_product_valid.to_csv(savepath, index_label=False)
    return df_user_product_valid


def make_user_product_prior(savepath, df_orders, df_order_products_prior):
    # select eval set == prior
    df_order_user_prior = df_orders.loc[df_orders.eval_set == "prior"]
    # only select order_id and user_id
    df_order_user_prior = df_order_user_prior[["order_id", "user_id"]]
    # merge order_user with order_products
    df_merged = pd.merge(df_order_user_prior, df_order_products_prior[["order_id", "product_id"]], on="order_id")
    # select user_id and product_id only
    df_user_product_prior = df_merged[["user_id", "product_id"]]
    # group the same user_id and product_id
    df_user_product_prior = df_user_product_prior.groupby(["user_id", "product_id"]).size().reset_index().rename(
        columns={0: "quantity"})
    df_user_product_prior.to_csv(savepath, index_label=False)
    return df_user_product_prior


def get_user_product_matrix(savepath, df_user_product_prior):
    user_product_matrix = sparse.coo_matrix((df_user_product_prior["quantity"],
                                             (df_user_product_prior["product_id"].cat.codes.copy(),
                                              df_user_product_prior["user_id"].cat.codes.copy())))
    sparse.save_npz(savepath, user_product_matrix)
    return user_product_matrix


def spars(matrix):
    total_size = matrix.shape[0] * matrix.shape[1]
    actual_size = matrix.size
    sparsity = (1 - (actual_size / total_size)) * 100
    return sparsity


def main():
    # Order datasets
    df_order_product_prior = pd.read_csv("order_products__prior.csv")
    df_order_product_train = pd.read_csv("order_products__train.csv")
    df_orders = pd.read_csv("orders.csv")
    print('orders shape', df_orders.shape)

    # Products
    df_products = pd.read_csv("products.csv")

    # Merge prior orders and products
    df_merged_order_products_prior = pd.merge(df_order_product_prior, df_products, on="product_id", how="left")

    # make user_product validation set with number of reorder matrix
    valid_exist = True
    if valid_exist:
        df_user_product_valid = pd.read_csv("user_product_valid.csv")
    else:
        df_user_product_valid = make_valid_set("user_product_valid.csv",
                                               df_orders, df_order_product_train)
    print('user_product_valid is done')
    print('valid shape', df_user_product_valid.shape)

    # make user_product train set with number of reorder matrix
    train_exist = True
    if train_exist:
        df_user_product_train = pd.read_csv("user_product_prior.csv").astype('category')
    else:
        df_user_product_train = make_user_product_prior('user_product_prior.csv',
                                                        df_orders, df_order_product_prior).astype("category")
    print('user_product_train is done')
    print('train shape', df_user_product_train.shape)

    # make utility matrix
    matrix_exist = True
    if matrix_exist:
        user_product_matrix = sparse.load_npz("product_user_matrix.npz").tocsr().astype(np.float32)
    else:
        user_product_matrix = get_user_product_matrix("product_user_matrix.npz", df_user_product_train)
    print('utility matrix is done')
    print('sparsity of the utility matrix is', spars(user_product_matrix))

if __name__ == "__main__":
    main()
