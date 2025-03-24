-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS user_database;
USE user_database;

-- Create users table if it doesn't exist (preserve existing)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    profile_image LONGBLOB
);

-- Create health_data table if it doesn't exist (preserve existing)
CREATE TABLE IF NOT EXISTS health_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    height FLOAT NOT NULL,
    weight FLOAT NOT NULL,
    bmi FLOAT NOT NULL,
    age INT NOT NULL,
    diabetes ENUM('none', 'type1', 'type2') DEFAULT 'none',
    bp ENUM('normal', 'low', 'high') DEFAULT 'normal',
    cholesterol ENUM('normal', 'low', 'mid', 'high') DEFAULT 'normal',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Drop existing product-related tables if they exist
DROP TABLE IF EXISTS product_tags;
DROP TABLE IF EXISTS product_health_restrictions;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS health_filters;

-- Create categories table
CREATE TABLE categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

-- Create products table
CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    weight VARCHAR(20) NOT NULL,
    image_url VARCHAR(255) NOT NULL,
    fat DECIMAL(5,2) DEFAULT 0,
    sugars DECIMAL(5,2) DEFAULT 0,
    sodium DECIMAL(5,2) DEFAULT 0,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
);

-- Create health_filters table
CREATE TABLE health_filters (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

-- Create product_health_restrictions table
CREATE TABLE product_health_restrictions (
    product_id INT NOT NULL,
    filter_id INT NOT NULL,
    PRIMARY KEY (product_id, filter_id),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
    FOREIGN KEY (filter_id) REFERENCES health_filters(id) ON DELETE CASCADE
);

-- Create product_tags table
CREATE TABLE product_tags (
    product_id INT NOT NULL,
    tag VARCHAR(20) NOT NULL,
    PRIMARY KEY (product_id, tag),
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
);

-- Insert default categories
INSERT INTO categories (name) VALUES 
('snacks'),
('breakfast-cereals'),
('chocolates'),
('beverages'),
('dairy'),
('instant-foods'),
('groceries'),
('food-supplements');

-- Insert default health filters
INSERT INTO health_filters (name) VALUES 
('diabetes'),
('heart'),
('obesity'),
('hypertension');

-- Useful Queries Section
-- =====================

-- 1. View table structures
DESC products;
DESC product_tags;
DESC product_health_restrictions;
DESC categories;
DESC health_filters;

-- 2. View all data in tables
SELECT * FROM products;
SELECT * FROM product_health_restrictions;
SELECT * FROM product_tags;
SELECT * FROM categories;
SELECT * FROM health_filters;

-- 3. View complete product information with categories, health restrictions, and tags
SELECT 
    p.id,
    p.name as product_name,
    p.price,
    p.weight,
    p.image_url,
    p.fat,
    p.sugars,
    p.sodium,
    c.name as category_name,
    GROUP_CONCAT(DISTINCT hf.name) as health_restrictions,
    GROUP_CONCAT(DISTINCT pt.tag) as tags
FROM products p 
JOIN categories c ON p.category_id = c.id
LEFT JOIN product_health_restrictions phr ON p.id = phr.product_id
LEFT JOIN health_filters hf ON phr.filter_id = hf.id
LEFT JOIN product_tags pt ON p.id = pt.product_id
GROUP BY p.id;

-- 4. View products by category
SELECT 
    c.name as category_name,
    COUNT(p.id) as product_count,
    GROUP_CONCAT(p.name) as products
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
GROUP BY c.id, c.name;

-- 5. View health restrictions by product
SELECT 
    p.name as product_name,
    GROUP_CONCAT(hf.name) as health_restrictions
FROM products p
LEFT JOIN product_health_restrictions phr ON p.id = phr.product_id
LEFT JOIN health_filters hf ON phr.filter_id = hf.id
GROUP BY p.id, p.name;

-- 6. View products with high fat content
SELECT 
    p.name,
    p.fat,
    c.name as category_name
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE p.fat > 10
ORDER BY p.fat DESC;

-- 7. View products with high sugar content
SELECT 
    p.name,
    p.sugars,
    c.name as category_name
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE p.sugars > 10
ORDER BY p.sugars DESC;

-- 8. View products with high sodium content
SELECT 
    p.name,
    p.sodium,
    c.name as category_name
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE p.sodium > 500
ORDER BY p.sodium DESC; 