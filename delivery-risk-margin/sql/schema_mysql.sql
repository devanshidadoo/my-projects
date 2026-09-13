-- deliveryrisk operational schema (mysql); 9 tables

-- customers: One row per shipping identity. `customer_unique_id` links repeat buyers.
CREATE TABLE IF NOT EXISTS customers (
  customer_id VARCHAR(40) NOT NULL,
  customer_unique_id VARCHAR(40) NOT NULL COMMENT 'stable across orders',
  customer_state VARCHAR(64) NOT NULL,
  customer_zip_prefix INT NOT NULL,
  first_seen_ts DOUBLE NOT NULL COMMENT 'acquisition time; safe as a feature',
  PRIMARY KEY (customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_customers_customer_unique_id ON customers (customer_unique_id);

-- sellers: Merchants. Handling time is a seller property and the largest controllable delay term.
CREATE TABLE IF NOT EXISTS sellers (
  seller_id VARCHAR(40) NOT NULL,
  seller_state VARCHAR(64) NOT NULL,
  seller_zip_prefix INT NOT NULL,
  onboarded_ts DOUBLE NOT NULL,
  fulfilment_mode VARCHAR(64) NOT NULL COMMENT 'merchant | warehouse',
  PRIMARY KEY (seller_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_sellers_seller_state ON sellers (seller_state);

-- products: Catalogue. Dimensions drive freight class, which drives lane choice.
CREATE TABLE IF NOT EXISTS products (
  product_id VARCHAR(40) NOT NULL,
  category VARCHAR(64) NOT NULL,
  weight_g DOUBLE NOT NULL,
  length_cm DOUBLE NOT NULL,
  height_cm DOUBLE NOT NULL,
  width_cm DOUBLE NOT NULL,
  PRIMARY KEY (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_products_category ON products (category);

-- carriers: Service levels, and what it costs to buy a faster one on the day.
CREATE TABLE IF NOT EXISTS carriers (
  carrier_id VARCHAR(40) NOT NULL,
  carrier_name VARCHAR(64) NOT NULL,
  service_level VARCHAR(64) NOT NULL COMMENT 'economy | standard | express',
  reliability_index DOUBLE NOT NULL COMMENT 'long-run share of shipments inside promise',
  expedite_surcharge DOUBLE NOT NULL COMMENT 'currency cost of upgrading one parcel',
  daily_capacity INT NOT NULL,
  PRIMARY KEY (carrier_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- shipping_lanes: Origin state x destination state x carrier. The lane is the unit transit time is a property of; a seller's late rate is not portable across lanes.
CREATE TABLE IF NOT EXISTS shipping_lanes (
  lane_id VARCHAR(40) NOT NULL,
  origin_state VARCHAR(64) NOT NULL,
  dest_state VARCHAR(64) NOT NULL,
  carrier_id VARCHAR(40) NOT NULL,
  distance_km DOUBLE NOT NULL,
  base_transit_days DOUBLE NOT NULL COMMENT 'carrier's published transit time',
  PRIMARY KEY (lane_id),
  FOREIGN KEY (carrier_id) REFERENCES carriers(carrier_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_shipping_lanes_carrier_id ON shipping_lanes (carrier_id);
CREATE INDEX ix_shipping_lanes_origin_state_dest_state ON shipping_lanes (origin_state, dest_state);

-- orders: The order header. `approved_ts` is the decision point; `pickup_ts` and `delivered_ts` are outcomes of this order and must never reach its own feature row.
CREATE TABLE IF NOT EXISTS orders (
  order_id VARCHAR(40) NOT NULL,
  customer_id VARCHAR(40) NOT NULL,
  order_status VARCHAR(64) NOT NULL COMMENT 'delivered | shipped | cancelled',
  purchase_ts DOUBLE NOT NULL,
  approved_ts DOUBLE COMMENT 'DECISION POINT',
  pickup_ts DOUBLE COMMENT 'carrier collection; outcome',
  delivered_ts DOUBLE COMMENT 'outcome',
  estimated_delivery_ts DOUBLE NOT NULL COMMENT 'the promise made at purchase',
  PRIMARY KEY (order_id),
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_orders_approved_ts ON orders (approved_ts);
CREATE INDEX ix_orders_customer_id ON orders (customer_id);
CREATE INDEX ix_orders_delivered_ts ON orders (delivered_ts);

-- order_items: One row per line. An order with two sellers has two handling clocks, not one.
CREATE TABLE IF NOT EXISTS order_items (
  order_id VARCHAR(40) NOT NULL,
  item_seq INT NOT NULL,
  product_id VARCHAR(40) NOT NULL,
  seller_id VARCHAR(40) NOT NULL,
  lane_id VARCHAR(40) NOT NULL COMMENT 'resolved shipping lane for this line',
  shipping_limit_ts DOUBLE NOT NULL COMMENT 'contractual handover deadline for the seller',
  price DOUBLE NOT NULL,
  freight_value DOUBLE NOT NULL,
  PRIMARY KEY (order_id, item_seq),
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id),
  FOREIGN KEY (seller_id) REFERENCES sellers(seller_id),
  FOREIGN KEY (lane_id) REFERENCES shipping_lanes(lane_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_order_items_seller_id ON order_items (seller_id);
CREATE INDEX ix_order_items_lane_id ON order_items (lane_id);
CREATE INDEX ix_order_items_product_id ON order_items (product_id);

-- order_payments: Payment legs. Installment counts and payment type both shift approval latency.
CREATE TABLE IF NOT EXISTS order_payments (
  order_id VARCHAR(40) NOT NULL,
  payment_seq INT NOT NULL,
  payment_type VARCHAR(64) NOT NULL,
  installments INT NOT NULL,
  payment_value DOUBLE NOT NULL,
  PRIMARY KEY (order_id, payment_seq),
  FOREIGN KEY (order_id) REFERENCES orders(order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- order_reviews: Post-delivery satisfaction. Used only to price the cost of a late delivery (policy/economics.py); it is structurally post-decision and is never a feature.
CREATE TABLE IF NOT EXISTS order_reviews (
  review_id VARCHAR(40) NOT NULL,
  order_id VARCHAR(40) NOT NULL,
  review_score INT NOT NULL,
  review_creation_ts DOUBLE NOT NULL,
  review_answer_ts DOUBLE,
  PRIMARY KEY (review_id),
  FOREIGN KEY (order_id) REFERENCES orders(order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_order_reviews_order_id ON order_reviews (order_id);
