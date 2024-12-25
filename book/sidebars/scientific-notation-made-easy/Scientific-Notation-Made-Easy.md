# Intuitive way of understanding the meaning of "7e-6" in Python


The letter "e" in Python stands for "exponent," and it's used to simplify the expression of numbers in scientific notation. This is particularly common in programming languages and calculators for representing very large or very small numbers.

For instance, "3 * 10^2" can be written in scientific notation as "3e2," and "5 * 10^-3" can be expressed as "5e-3."

```python
import math

print(3e2) # 3 * 10^2 = 300
print(5e-3) # 5 * 10^-3 = 0.005

x = 3e6 
y = 2e4
z = x / y # 3,000,000 / 20,000 = 150

a = math.sqrt(2e3) # square root of 2 * 1000
print(a) # 44.7213595499958
```

Using "e" in this manner allows for quick and accurate input or reading of numbers. It's especially useful in fields like computer science and data analysis.

Here's a simple trick to easily convert these notations: When you see a number like "7e-6," you can think of the number after "e" as the total count of zeros you'll have. In this case, the "-6" tells you that you're moving the decimal point 6 places to the left. So, you end up with 1 zero before the decimal point and 5 zeros after the decimal point, totaling 6 zeros. This gives you the number 0.000007. Just count the number of zeros. 

So the trick is to look at the number after "e" as the total count of zeros you'll have when you write out the number in standard form.

Conversely, if you see "7e6," it means "7 times ten to the power of 6," or "7 * 10^6."

Since there's no decimal point, you can simply append six zeros to the "7," making it "7,000,000."

Easy peasy.

Here's a pop quiz. Convert this learning_rate to its standard form: 

```python

learning_rate=1e-6 # 0.000001

```

Yes, you need 6 zeros. 1 x 10^-6 = 0.000001.

‼️ Note: mathematically and scientifically speaking, the number after "e" does not represent the total count of zeros you'll have when you write out the number in standard form. Instead, it represents the number of places you move the decimal point. However, it's way easier and more intuitive for us 'humans' just to count the number of zeros. 