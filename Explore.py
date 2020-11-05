import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats



def get_user_product_prior(df_orders, df_order_products_prior):
    """
    Generates a dataframe of users and their prior products purchases
    [user id & product id & quantity]
    """
    print("Creating prior user-product data frame ...")

    # Consider ony "prior" orders and remove all columns except `user_id` from `df_orders`
    df_order_user_prior = df_orders.loc[df_orders.eval_set == "prior"]
    df_order_user_prior = df_order_user_prior[["order_id", "user_id"]]

    # Remove all columns except order_id and user_id from df_orders and
    # merge the above on `order_id` and remove `order_id`
    df_merged = pd.merge(df_order_user_prior, df_order_products_prior[["order_id", "product_id"]], on="order_id")
    user_product_prior  = df_merged[["user_id", "product_id"]]
    df_user_product_prior = user_product_prior.groupby(["user_id", "product_id"]).size().reset_index().rename(
        columns={0: "quantity"})

    user_product_prior_raw = df_order_products_prior.groupby("order_id")["product_id"].apply(list).reset_index().rename(
        columns={"product_id": "products"})
    df_user_product_prior_raw = pd.merge(df_order_user_prior, user_product_prior_raw, on="order_id")
    df_user_product_prior_raw = df_user_product_prior_raw[["user_id", "products"]]

    # Write to disk
    df_user_product_prior_raw.to_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/user_product_prior_raw.csv",
        index_label=False)
    df_user_product_prior.to_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/user_product_prior.csv",
        index_label=False)
    return df_user_product_prior_raw, df_user_product_prior


def make_validation_data(df_orders, df_order_products_train):
    """
    Generates the validation dataset
    [user id & products id(array)]
    """
    print("Creating validation data ...")

    # Read train csv
    df_order_user_current = df_orders.loc[(df_orders.eval_set == "train")].reset_index()
    df_order_user_current = df_order_user_current[["order_id", "user_id"]]

    # Convert train dataframe to a similar format
    df_order_products_test = df_order_products_train[["order_id", "product_id"]]
    df_order_products_test = df_order_products_test.groupby("order_id")["product_id"].apply(list).reset_index().rename(
        columns={"product_id": "products"})

    # Merge on order id
    df_user_products_validation = pd.merge(df_order_user_current, df_order_products_test, on="order_id")
    df_user_products_validation = df_user_products_validation[["user_id", "products"]]

    df_user_products_validation.to_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/user_products_validation.csv"
        , index_label=False)

if __name__ == "__main__":
    # importing datasets
    aisles = pd.read_csv("/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/aisles.csv")
    departments = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/departments.csv")
    orders = pd.read_csv("/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/orders.csv")
    order_products_prior = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/order_products__prior.csv")
    order_products_train = pd.read_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/order_products__train.csv")
    products = pd.read_csv("/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/products.csv")

    # Concatenation of prior & train to order_products_total
    order_products_total = pd.concat([order_products_prior, order_products_train])
    # amount of orders distribution
    orders_amount_per_customer = orders.groupby('user_id')['order_number'].count().value_counts()
    # Merging aisles, departments & products
    products_departments = products.merge(departments, left_on='department_id', right_on='department_id', how='left')
    products_departments_aisles = products_departments.merge(aisles, left_on='aisle_id', right_on='aisle_id',
                                                             how='left')
    # Merging products_departments_aisles and order_products_total.
    df = order_products_total.merge(products_departments_aisles, left_on='product_id', right_on='product_id',
                                    how='left')

    # General datasets info
    print('there are', orders.shape[0], "orders in total")
    print('there are', len(orders[orders.eval_set == 'prior']), 'entries for prior')
    print('there are', len(orders[orders.eval_set == 'train']), 'entries for train')
    print('there are', len(orders[orders.eval_set == 'test']), 'entries for test')

    print('there are', len(orders[orders.eval_set == 'prior'].user_id.unique()), 'unique customers in total')
    print('there are', len(orders[orders.eval_set == 'train'].user_id.unique()), 'unique customers in train set')
    print('there are', len(orders[orders.eval_set == 'test'].user_id.unique()), 'unique customers in test set')

    # merge datasets
    df_user_product_prior_raw, user_product_prior = get_user_product_prior(orders, order_products_prior)
    user_products_validation = make_validation_data(orders, order_products_train)
    order_products_prior = pd.merge(order_products_prior, df_products, on="product_id", how="left")
    order_products_prior.to_csv(
        "/Users/dona/Desktop/Spring 2020/CS 229/229 Project/instacart_2017_05_01_data/order_products_prior.csv"
        , index_label=False)




