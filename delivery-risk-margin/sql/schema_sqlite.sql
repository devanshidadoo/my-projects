-- deliveryrisk operational schema (sqlite); 9 tables

-- customers: One row per shipping identity. `customer_unique_id` links repeat buyers.
CREATE TABLE IF NOT EXISTS customers (
  customer_id TEXT NOT NULL,
  customer_unique_id TEXT NOT NULL,
  customer_state TEXT NOT NULL,
  customer_zip_prefix INTEGER NOT NULL,
  first_seen_ts REAL NOT NULL,
  PRIMARY KEY (customer_id)
);
CREATE INDEX IF NOT EXISTS ix_customers_customer_unique_id ON customers (customer_unique_id);

-- sellers: Merchants. Handling time is a seller property and the largest controllable delay term.
CREATE TABLE IF NOT EXISTS sellers (
  seller_id TEXT NOT NULL,
  seller_state TEXT NOT NULL,
  seller_zip_prefix INTEGER NOT NULL,
  onboarded_ts REAL NOT NULL,
  fulfilment_mode TEXT NOT NULL,
  PRIMARY KEY (seller_id)
);
CREATE INDEX IF NOT EXISTS ix_sellers_seller_state ON sellers (seller_state);

-- products: Catalogue. Dimensions drive freight class, which drives lane choice.
CREATE TABLE IF NOT EXISTS products (
  product_id TEXT NOT NULL,
  category TEXT NOT NULL,
  weight_g REAL NOT NULL,
  length_cm REAL NOT NULL,
  height_cm REAL NOT NULL,
  width_cm REAL NOT NULL,
  PRIMARY KEY (product_id)
);
CREATE INDEX IF NOT EXISTS ix_products_category ON products (category);

-- carriers: Service levels, and what it costs to buy a faster one on the day.
CREATE TABLE IF NOT EXISTS carriers (
  carrier_id TEXT NOT NULL,
  carrier_name TEXT NOT NULL,
  service_level TEXT NOT NULL,
  reliability_index REAL NOT NULL,
  expedite_surcharge REAL NOT NULL,
  daily_capacity INTEGER NOT NULL,
  PRIMARY KEY (carrier_id)
);

-- shipping_lanes: Origin state x destination state x carrier. The lane is the unit transit time is a property of; a seller's late rate is not portable across lanes.
CREATE TABLE IF NOT EXISTS shipping_lanes (
  lane_id TEXT NOT NULL,
  origin_state TEXT NOT NULL,
  dest_state TEXT NOT NULL,
  carrier_id TEXT NOT NULL,
  distance_km REAL NOT NULL,
  base_transit_days REAL NOT NULL,
  PRIMARY KEY (lane_id),
  FOREIGN KEY (carrier_id) REFERENCES carriers(carrier_id)
);
CREATE INDEX IF NOT EXISTS ix_shipping_lanes_carrier_id ON shipping_lanes (carrier_id);
CREATE INDEX IF NOT EXISTS ix_shipping_lanes_origin_state_dest_state ON shipping_lanes (origin_state, dest_state);

-- orders: The order header. `approved_ts` is the decision point; `pickup_ts` and `delivered_ts` are outcomes of this order and must never reach its own feature row.
CREATE TABLE IF NOT EXISTS orders (
  order_id TEXT NOT NULL,
  customer_id TEXT NOT NULL,
  order_status TEXT NOT NULL,
  purchase_ts REAL NOT NULL,
  approved_ts REAL,
  pickup_ts REAL,
  delivered_ts REAL,
  estimated_delivery_ts REAL NOT NULL,
  PRIMARY KEY (order_id),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE INDEX IF NOT EXISTS ix_orders_approved_ts ON orders (approved_ts);
CREATE INDEX IF NOT EXISTS ix_orders_customer_id ON orders (customer_id);
CREATE INDEX IF NOT EXISTS ix_orders_delivered_ts ON orders (delivered_ts);

-- order_items: One row per line. An order with two sellers has two handling clocks, not one.
CREATE TABLE IF NOT EXISTS order_items (
  order_id TEXT NOT NULL,
  item_seq INTEGER NOT NULL,
  product_id TEXT NOT NULL,
  seller_id TEXT NOT NULL,
  lane_id TEXT NOT NULL,
  shipping_limit_ts REAL NOT NULL,
  price REAL NOT NULL,
  freight_value REAL NOT NULL,
  PRIMARY KEY (order_id, item_seq),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id),
  FOREIGN KEY (seller_id) REFERENCES sellers(seller_id),
  FOREIGN KEY (lane_id) REFERENCES shipping_lanes(lane_id)
);
CREATE INDEX IF NOT EXISTS ix_order_items_seller_id ON order_items (seller_id);
CREATE INDEX IF NOT EXISTS ix_order_items_lane_id ON order_items (lane_id);
CREATE INDEX IF NOT EXISTS ix_order_items_product_id ON order_items (product_id);

-- order_payments: Payment legs. Installment counts and payment type both shift approval latency.
CREATE TABLE IF NOT EXISTS order_payments (
  order_id TEXT NOT NULL,
  payment_seq INTEGER NOT NULL,
  payment_type TEXT NOT NULL,
  installments INTEGER NOT NULL,
  payment_value REAL NOT NULL,
  PRIMARY KEY (order_id, payment_seq),
  FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- order_reviews: Post-delivery satisfaction. Used only to price the cost of a late delivery (policy/economics.py); it is structurally post-decision and is never a feature.
CREATE TABLE IF NOT EXISTS order_reviews (
  review_id TEXT NOT NULL,
  order_id TEXT NOT NULL,
  review_score INTEGER NOT NULL,
  review_creation_ts REAL NOT NULL,
  review_answer_ts REAL,
  PRIMARY KEY (review_id),
  FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
CREATE INDEX IF NOT EXISTS ix_order_reviews_order_id ON order_reviews (order_id);
