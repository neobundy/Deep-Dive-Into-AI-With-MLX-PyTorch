# Chapter 11 - Tenny, the Stock Classifier

![pytorch.png](images%2Fpytorch.png)

If you're looking to evolve Tenny from an Analyst to a Classifier, the initial step is to modify the neural network architecture to be more suited for classification tasks. However, there's no need to completely reinvent the wheel. We already have a functioning neural network architecture designed for regression and other functions. Why not adapt and reuse this existing framework?

Isn't the concept of object orientation fascinating? But don't limit yourself to just the realm of coding. Consider extending this idea beyond programming. Imagine everything as an object, including your thoughts, emotions, actions, and even your entire self. With this perspective, how might you approach designing or redesigning your life?

## A Little Object-Oriented Cooking Secret - What the Heck?

![object-oriented-kitchen.png](..%2F..%2Fessays%2Flife%2Fimages%2Fobject-oriented-kitchen.png)

What's your take on cooking?

As a self-proclaimed chef in my kitchen, I can go on about cooking for hours. But here's something you might not often hear:

Cooking, like everything else in our universe, is an object.

Yes, you read that right.

Moreover, everything related to cooking, whether tangible or intangible, is an object. This idea stems from the broader concept:

> Everything, tangible or intangible, is an object.

So, why not embrace it?

As I've emphasized up to this point, the beauty of object-oriented programming (OOP) goes way beyond coding. It offers a unique way to view the world. In OOP, we perceive everything as an object with attributes and behaviors. This perspective is transformative, even in everyday life.

When you think about complex things like cooking and taste, how do you simplify them? By extracting commonalities as abstract attributes or methods.

Why do we cook? To eat, right? Therefore, the ultimate goal of cooking is to eat – and not just eat, but eat well. Delicious food is the end goal.

Ever wondered what makes food taste good? It's a mix of ingredients, the cooking process, and presentation. Ultimately, three factors really sway your opinion: fat, salt, and sugar – the greasiness, saltiness, and sweetness of your food.

I'm not just talking about abstract concepts; it's a scientific fact. These are the three pillars of taste. For more insights, check out Michael Moss's book, "Salt Sugar Fat: How the Food Giants Hooked Us."

https://www.amazon.com/Salt-Sugar-Fat-Giants-Hooked/dp/0812982193/

Your food's appearance and aroma matter, but if you miss any of these three taste pillars, it won't be as delicious as it could be. And even with all three, balance is key. Too much or too little of anything is not good.

Each ingredient contributes to the taste. Onions, for instance, are naturally sweet. Seafood like fish, shrimp, and crab are salty by nature, and so forth.

Recipes? I don't memorize them. When I taste a dish, I try to discern what ingredients contribute to its flavor and in what proportions. Then, I can recreate or even improve the dish.

Why commit recipes to memory? That approach isn't object-oriented. It falls short in terms of scalability, maintainability, reusability, flexibility, extensibility, elegance, beauty, enjoyment, and creativity. It's not in the spirit of object-orientation, after all! Plus, everyone's taste preferences vary. So, why not craft your own recipes by adapting and modifying existing ones through inheritance and polymorphism?

Here's another insight about recipes: they're essentially referencing the same concepts, just presented in different ways. It's a classic case of inheritance and polymorphism in action. That's really all there is to it. 

But, being Korean, I add a bit more magic: Umami. Japanese and Koreans share this magical element of flavor explosion. Consider kelp or dried bonito flakes, rich sources of Umami. We use them in broths for soups and stews. 

Even if you're from an English-speaking area, you probably enjoy Ramen (Japanese) or Ramyon (Korean) noodles, right? Their exceptional taste comes from Umami – the Japanese and Korean secret weapon of flavor. In Korean, we call it '감칠맛'. There's no direct translation in English. It's a blend of many elements. Although we can tell if a dish has umami, it's hard to pinpoint exactly what it is. Umami is the official term for this taste. MSG is an artificial version of Umami. We've reduced MSG usage nowadays, but Ramen or Ramyon noodles used to be loaded with it. That's why they taste so good. Even now, they contain the right mix of ingredients to achieve that Umami flavor.

So, why do these dishes taste so good? Think of it from an object-oriented perspective. They meet the four pillars of great taste: fat, salt, sugar, and umami. You already have a solid base class of taste to derive from to understand good flavor.

Yes, Ramyon tastes fantastic because it balances the right amounts of fat, salt, sugar, and umami.

What about pizza? It's delicious for the same reason: a perfect blend of fat, salt, sugar, and umami.

Why do we add salt to steak? To enhance its flavor. We're aiming for the right balance of fat, salt, and sugar. And umami? Well, that might not be a typical western addition, but marinated steak with soy sauce, adding that umami flavor, is something many enjoy.

Korean Bulgogi(불고기) and Galbi(갈비) follow the same culinary principles: achieving harmony between fat, salt, sugar, and umami. This rule is fundamental to every Korean or Japanese dish, each with its unique application of polymorphism. Every dish encompasses the four essential pillars of taste, neatly encapsulated. Complex dishes inherit from simpler ones, which in turn stem from these fundamental taste pillars. It's simple, elegant, and beautiful.

Korean Kimchi exemplifies the harmonious blend of the four pillars of taste. As a fermented dish, its natural sourness complements its saltiness, sweetness, and spiciness. Skillfully blending fat, salt, sugar, and umami, Kimchi has evolved into a culinary masterpiece over thousands of years. And remember, spiciness isn't a taste but a sensation. It might seem counter-intuitive, but spiciness is akin to a type of enjoyable pain, almost like a self-imposed culinary torture. 

That's the essence of cooking. Herbs and other additional ingredients are merely variations on these basic taste pillars. They're the spices of life, nothing more. To make your food taste great, understand these four taste pillars and abstract them into a generic class. Then, derive from it to interpret dishes with or without recipes, or
create your own dish with or without recipes. 

Honestly, I'm always reluctant when someone asks me how to cook what I've prepared for them. Without grasping this fundamental principle, replicating the flavor is challenging. So, I usually offer to cook for them if they visit and I have the time. Besides, I don't really ponder over recipes unless specifically asked to do so.

Now, try explaining why your favorite food tastes good and why others might not, setting aside personal preferences. Be generic and abstract. You'll be amazed at what you can learn from this exercise.

This approach offers a unique way to view our universe through the lens of object-orientation: the ultimate life hack.

Here's another fun experiment. Inform your GPT model about this perspective on taste. Then, ask it to explain why a dish tastes good from the same viewpoint. It's a one-shot learning experience for it, but it will be capable of interpreting the taste of any dish from an object-oriented perspective. You should be able to do the same. I mean it

## Key Concepts to Consider in Transforming Tenny into a Classifier

We have a lot to cover in this chapter in order to transform Tenny into a classifier. Let's start with the key concepts to consider.

To upgrade Tenny from a regression model to a classifier, we need to adjust the architecture and output layer of the neural network to suit classification tasks. In classification models, the output layer typically uses a softmax function for multi-class classification or a sigmoid function for binary classification. This is different from regression models, where the output layer is usually a single neuron without an activation function for continuous value prediction.

Let's define `TennyClassifier` inheriting from `Tenny` and modify it for classification:

```python
class TennyClassifier(Tenny):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TennyClassifier, self).__init__(input_size, hidden_size, num_classes)
        # The final fully connected layer's output size is set based on `num_classes`
        self.fc5 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)  # Applying softmax to get probabilities
```

We will be looking into TennyClassifier in more detail in the next section, but for now, let's focus on activation functions.

### Sigmoid vs. Softmax

In classification tasks, `softmax` and `sigmoid` functions play crucial roles as activation functions, especially in the output layer of neural networks. They help to interpret the outputs of the model as probabilities, making the model's predictions more interpretable.

### Softmax Function:
- **Use Case**: Softmax is used in multi-class classification tasks where an instance can belong to one of many classes.
- **Functionality**: It converts the output scores (also known as logits) from the model into probabilities. The probabilities sum up to 1, making it a good fit for representing a probability distribution across multiple classes.
- **Example**: If a model predicts `[2.0, 1.0, 0.1]` for three classes, softmax will convert these values into probabilities.

```python
import torch.nn.functional as F

logits = [2.0, 1.0, 0.1]
probabilities = F.softmax(torch.tensor(logits), dim=0)
print(probabilities)
```

### Sigmoid Function:
- **Use Case**: Sigmoid is typically used in binary classification tasks where an instance belongs to one of two classes.
- **Functionality**: It maps the model's output, which can be any real value, into a probability between 0 and 1. This makes it suitable for predicting the probability of the positive class in binary classification.
- **Example**: If a model predicts `1.5` for a binary class, sigmoid will convert this value into a probability.

```python
import torch

logit = torch.tensor([1.5])
probability = torch.sigmoid(logit)
print(probability)
```

### Key Differences:
- **Softmax** is suitable for cases where each instance should be classified into one of many classes (e.g., cat, dog, bird).
- **Sigmoid** is used when each instance is classified into one of two classes (e.g., spam or not spam).

### Importance in Classification:
1. **Interpretability**: These functions convert raw model outputs into probabilities, making the results interpretable (e.g., "There's an 80% chance that this image is of a cat").

2. **Training Stability**: They help in stabilizing the training process by bounding the output values, which can improve the numerical properties of the optimization problem.

3. **Loss Function Compatibility**: They align with loss functions used in classification (e.g., Cross-Entropy Loss) that expect probabilities as input.

In summary, softmax and sigmoid functions are integral to classification tasks in machine learning, as they provide a probabilistic interpretation of the model's raw outputs, thereby aiding in both the training and evaluation of the model.

Assess your understanding by exploring the following essay. Have you ever considered applying the softmax function to your life? Yes, it's possible. 

[Softmax-of-Life.md](..%2F..%2Fessays%2Flife%2FSoftmax-of-Life.md)

### Why Softmax is Suitable for Tenny with Three Classes

When dealing with a multi-class classification problem, where an instance is to be classified into one of several distinct classes, the softmax function is an ideal choice. This is precisely the case with our Tenny model, which has three classes: 'Growth', 'Stalwart', and 'Other'. In such scenarios, softmax is used in the output layer of the neural network to interpret the model’s outputs as probabilities.

1. **Probabilistic Interpretation**: Softmax takes the raw output scores (logits) for each class and transforms them into a probability distribution. The output from softmax for each class is a probability between 0 and 1, and the sum of these probabilities for all classes equals 1.

2. **Multi-class Suitability**: Contrary to the sigmoid function, which excels in binary classification, softmax is adept at managing multiple classes. This is especially beneficial for your model that needs to distinguish between three distinct categories. In a scenario of binary classification, where the choices are 'Growth' or 'Not Growth', the sigmoid function would be more appropriate. 

3. **Maximizing the Probability of the Correct Class**: In training, softmax works well with loss functions like cross-entropy, which aim to maximize the predicted probability of the correct class. This alignment is beneficial for effectively training the model.

Suppose Tenny’s model outputs raw logits for a particular instance as `[1.2, 0.9, 0.3]` corresponding to the classes 'Growth', 'Stalwart', and 'Other', respectively. Applying softmax to these logits would convert them into a probability distribution:

```python
import torch.nn.functional as F
import torch

logits = torch.tensor([1.2, 0.9, 0.3])
probabilities = F.softmax(logits, dim=0)
print(probabilities)
```

This code will output three values, each representing the probability that the given instance belongs to one of the three classes. The highest probability indicates the model’s predicted class.

In the case of Tenny, since we have more than two classes, softmax is the appropriate choice for the output layer. It ensures that the output of our model is a valid probability distribution over the three classes, aiding in both the interpretability and effectiveness of the model in classifying instances into the categories of 'Growth', 'Stalwart', or 'Other'.

### Data Format for Classification

Let's look at the final format of our data:

```text
Fiscal Quarters,2Q FY2018,3Q FY2018,4Q FY2018,1Q FY2019,2Q FY2019,3Q FY2019,4Q FY2019,1Q FY2020,2Q FY2020,3Q FY2020,4Q FY2020,1Q FY2021,2Q FY2021,3Q FY2021,4Q FY2021,1Q FY2022,2Q FY2022,3Q FY2022,4Q FY2022,1Q FY2023,2Q FY2023,3Q FY2023,4Q FY2023
Normalized Price,42.64,47.816,55.8,39.168,50.531,52.569,61.295,79.884,74.151,96.944,116.512,143.22,134.7,147.805,154.224,160.855,165.54,158.6,145.755,151.887,166.662,191.958,178.64
Price / Earnings - P/E (LTM),16.4,17.2,18.6,12.8,16.9,17.7,20.5,25.2,23.1,29.2,35.2,38.5,30.0,28.7,27.2,26.5,26.7,26.0,23.7,25.7,28.2,32.1,29.0
Net EPS - Basic,2.6,2.78,3.0,3.06,2.99,2.97,2.99,3.17,3.21,3.32,3.31,3.72,4.49,5.15,5.67,6.07,6.2,6.1,6.15,5.91,5.91,5.98,6.16
Return On Equity %,0.4086,0.4537,0.4936,0.4605,0.4913,0.5269,0.5592,0.5547,0.6209,0.6925,0.7369,0.8209,1.034,1.2712,1.4744,1.4557,1.4927,1.6282,1.7546,1.4794,1.4561,1.6009,1.7195
Total Revenues / CAGR 5Y,0.0791,0.0855,0.0922,0.085,0.0799,0.0777,0.0731,0.0602,0.0478,0.0407,0.0327,0.0459,0.0742,0.0952,0.1115,0.1164,0.1186,0.1164,0.1146,0.1013,0.0925,0.0851,0.0761
Net Income / CAGR 5Y,0.0609,0.0825,0.0996,0.0992,0.0868,0.0763,0.0694,0.0529,0.0366,0.0286,0.0146,0.0354,0.0853,0.1267,0.1569,0.1733,0.1739,0.1639,0.156,0.135,0.1208,0.1105,0.1026
Normalized Net Income / CAGR 5Y,0.0505,0.0667,0.0777,0.0686,0.0577,0.0491,0.0421,0.0246,0.0068,-0.0017,-0.0154,0.0052,0.0553,0.0938,0.1222,0.1388,0.1412,0.1357,0.132,0.1122,0.1037,0.0965,0.093
Dividend Yield (LTM),0.0157,0.0146,0.013,0.0191,0.0154,0.015,0.0131,0.0101,0.011,0.0085,0.0072,0.0059,0.0064,0.0059,0.0058,0.0056,0.0056,0.0058,0.0064,0.0062,0.0057,0.0049,0.0054
Market Capitalization,835909.0,921558.7,1056653.0,731605.9,924543.7,946064.6,1080861.6,1392761.5,1270387.4,1648288.4,1957760.3,2389912.7,2228956.6,2430062.4,2506234.7,2600758.6,2652206.4,2532607.7,2308607.9,2388417.6,2606783.5,3024682.2,2761224.3
Beta,,,,,,,,,,,,,,,,,,,,,,,1.29
Industry Beta Average,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34
Industry PE Average,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25
Label,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart
```

Given the format of our CSV files for the `TennyClassifierDataset`, there are a few key considerations to ensure effective data processing:

#### Handling Quarter Identifiers

The first row contains quarter identifiers and is not a feature. This row should be used to set column names during data import but not included as part of the feature set. Since our `read_and_clean_data` method already handles the first row correctly by setting it as the index and then transposing the DataFrame, we're on the right track.

```python
    def read_and_clean_data(self, files):
        # Read all files at once
        # Transposing each DataFrame after reading it could be a costly operation. If possible, we need to change the format of the data files to avoid the need for transposition
        data = pd.concat([pd.read_csv(os.path.join(self.folder_path, file), index_col=0).transpose() for file in files], ignore_index=True)
        data = self.clean_data(data)
        return data
```

#### Features and Labels

Features range from the 2nd row (normalized price) to the 16th row (just before the 'Label' row). The 'Label' row is our target variable for classification. 

```python
class TennyClassifierDataset(TennyDataset):
    # [Other methods]

    def prepare_features_labels(self, df):
        # Assuming the last column is the relevant label
        labels = df.iloc[:, -1]  # Only the last label is used
        features = df.iloc[:, :-1]  # All columns except the last one

        # Convert labels to class indices
        labels = labels.map(self.one_hot_encode).values

        return features.values, labels

```
   - The method uses `iloc` for slicing the DataFrame to extract features.
   - The label is the last column in the DataFrame, so we use `iloc[:, -1]` to extract it.
   - The extracted features and labels are then converted to numpy arrays, which is a common format for feeding data into machine learning models.

I modified the code to specifically designate only the last column as the label.

#### Handling Constant Features

Features like 'Beta', 'Industry Beta Average', 'Industry PE Average' have the same values across all quarters, with only the last element being the most current and relevant. For these features, it's efficient to use only the most recent value (the last column in your dataset) rather than all historical values, as they don't add additional information.

The `TennyClassifierDataset` class, as an extension of `TennyDataset`, is a prime example of the elegance and power of Object-Oriented Programming (OOP). In OOP, the concept of inheritance allows a new class to receive, or "inherit", properties and methods from an existing class. This approach not only promotes code reuse but also enhances the maintainability and scalability of the code.

In the case of `TennyClassifierDataset`, it inherits from `TennyDataset`, meaning it automatically acquires all the functionalities of the parent class. This is evident in the `read_and_clean_data` method:

```python
class TennyClassifierDataset(TennyDataset):
    # Inherited __init__ and other methods from TennyDataset

    def read_and_clean_data(self, files):
        data = super().read_and_clean_data(files)

        # Select only the most recent value for constant features
        # After transposing, each of these features is a single-row dataframe or series.
        # We should take the value directly.
        data['Beta'] = data['Beta'].values[0]
        data['Industry Beta Average'] = data['Industry Beta Average'].values[0]
        data['Industry PE Average'] = data['Industry PE Average'].values[0]

        return data
```

## Analyzing the Tenny, the Classifier

The final example code is somewhat intricate, primarily because of the inheritance and polymorphism it incorporates. However, we successfully have Tenny, the Classifier model, up and running. To fully understand how it operates, it's crucial to analyze it in detail.

Ensure that you thoroughly read and comprehend the entire code before moving forward.

[tenny-the-analyst-v4-torch.py](tenny-the-analyst-v4-torch.py)

### Overview: How It Works

1. A folder named `enhanced-data-with-labels` which includes 12 CSV files. Each CSV file is labeled data for each of the 12 companies.

2. A folder named `tenny` which contains an `__init__.py` file, a Python package. As the name suggests, this folder contains the code for the Tenny base classes we inherit from.

3. The Python script `tenny-the-analyst-v4-torch.py`, which is the main codebase for the stock classification models.

A CSV file, `labeled-enhanced-raw-data-aapl.csv`, for example, which represents data for Apple (Apple), is structured as follows:


```text
Fiscal Quarters,2Q FY2018,3Q FY2018,4Q FY2018,1Q FY2019,2Q FY2019,3Q FY2019,4Q FY2019,1Q FY2020,2Q FY2020,3Q FY2020,4Q FY2020,1Q FY2021,2Q FY2021,3Q FY2021,4Q FY2021,1Q FY2022,2Q FY2022,3Q FY2022,4Q FY2022,1Q FY2023,2Q FY2023,3Q FY2023,4Q FY2023
Normalized Price,42.64,47.816,55.8,39.168,50.531,52.569,61.295,79.884,74.151,96.944,116.512,143.22,134.7,147.805,154.224,160.855,165.54,158.6,145.755,151.887,166.662,191.958,178.64
Price / Earnings - P/E (LTM),16.4,17.2,18.6,12.8,16.9,17.7,20.5,25.2,23.1,29.2,35.2,38.5,30.0,28.7,27.2,26.5,26.7,26.0,23.7,25.7,28.2,32.1,29.0
Net EPS - Basic,2.6,2.78,3.0,3.06,2.99,2.97,2.99,3.17,3.21,3.32,3.31,3.72,4.49,5.15,5.67,6.07,6.2,6.1,6.15,5.91,5.91,5.98,6.16
Return On Equity %,0.4086,0.4537,0.4936,0.4605,0.4913,0.5269,0.5592,0.5547,0.6209,0.6925,0.7369,0.8209,1.034,1.2712,1.4744,1.4557,1.4927,1.6282,1.7546,1.4794,1.4561,1.6009,1.7195
Total Revenues / CAGR 5Y,0.0791,0.0855,0.0922,0.085,0.0799,0.0777,0.0731,0.0602,0.0478,0.0407,0.0327,0.0459,0.0742,0.0952,0.1115,0.1164,0.1186,0.1164,0.1146,0.1013,0.0925,0.0851,0.0761
Net Income / CAGR 5Y,0.0609,0.0825,0.0996,0.0992,0.0868,0.0763,0.0694,0.0529,0.0366,0.0286,0.0146,0.0354,0.0853,0.1267,0.1569,0.1733,0.1739,0.1639,0.156,0.135,0.1208,0.1105,0.1026
Normalized Net Income / CAGR 5Y,0.0505,0.0667,0.0777,0.0686,0.0577,0.0491,0.0421,0.0246,0.0068,-0.0017,-0.0154,0.0052,0.0553,0.0938,0.1222,0.1388,0.1412,0.1357,0.132,0.1122,0.1037,0.0965,0.093
Dividend Yield (LTM),0.0157,0.0146,0.013,0.0191,0.0154,0.015,0.0131,0.0101,0.011,0.0085,0.0072,0.0059,0.0064,0.0059,0.0058,0.0056,0.0056,0.0058,0.0064,0.0062,0.0057,0.0049,0.0054
Market Capitalization,835909.0,921558.7,1056653.0,731605.9,924543.7,946064.6,1080861.6,1392761.5,1270387.4,1648288.4,1957760.3,2389912.7,2228956.6,2430062.4,2506234.7,2600758.6,2652206.4,2532607.7,2308607.9,2388417.6,2606783.5,3024682.2,2761224.3
Beta,,,,,,,,,,,,,,,,,,,,,,,1.29
Industry Beta Average,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34,1.34
Industry PE Average,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25,33.25
Label,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart,Stalwart

```

- The columns represent different fiscal quarters, 23 in total.
- Each row represent a different financial metric or indicator:
  - **Fiscal Quarters**: The fiscal quarters for which the data is reported. This is not a feature.
  - **Normalized Price**: The adjusted stock price accounting for corporate actions.
  - **Price/Earnings - P/E (LTM)**: The ratio comparing share price to earnings over the last twelve months.
  - **Net EPS - Basic**: The earnings per share based on net income and outstanding shares.
  - **Return On Equity %**: The percentage of profit relative to shareholders' equity.
  - **Total Revenues / CAGR 5Y**: The compound annual growth rate of total revenues over five years.
  - **Net Income / CAGR 5Y**: The growth rate of net income over a similar period.
  - **Normalized Net Income / CAGR 5Y**: The adjusted growth rate of net income over five years.
  - **Dividend Yield (LTM)**: The dividend yield over the last twelve months.
  - **Market Capitalization**: The market value of the company, calculated as the share price multiplied by the number of shares outstanding.
  - **Beta**: The beta value of the stock, which measures its volatility relative to the market.
  - **Industry Beta Average**: The average beta value of the industry.
  - **Industry PE Average**: The average price/earnings ratio of the industry.
  - **Label**: The target variable for classification. It has three possible values: Growth, Stalwart, and Other. Label is transformed into a one-hot encoded class index for classification: CLASS_INDICES = {GROWTH_STOCK: 0, STALWART_STOCK: 1, OTHER_STOCK: 2}
  
- Beta, Industry Beta Average, and Industry PE Average are constant features, meaning they have the same value across all quarters. For these features, we only need the most recent value, which is the last column in the CSV file.
- Some rows have missing values, which are represented by commas, and dealt with during data cleaning.
- The last row, 'Label', represents the target variable for classification. It has three possible values: Growth, Stalwart, and Other.
- All CSV files have a consistent format. Out of 12, one will be used for prediction, and the remaining 11 will be used for training and validation.

The CSV files within the `enhanced-data-with-labels` folder follow a uniform format. Each file presents data across 13 rows, representing various financial metrics, spread over multiple columns corresponding to fiscal quarters. However, there's an exception in `labeled-enhanced-raw-data-adbe.csv`, which uniquely features 25 columns, unlike the standard 24 in the others. This inconsistency is addressed in the code by ensuring uniformity in the number of columns across all files, with any additional columns being removed.

### `tenny` Package: Regression Model and Dataset Classes We Inherit From

The `tenny` package contains only an `__init__.py` file. This file is typically used to indicate that the directory should be considered a Python package and can also contain initialization code for the package.

The `__init__.py` file in the `tenny` package contains the implementation of several classes. This package is central to the model's architecture and data handling. Here's an overview of the classes defined in this file:

1. **TennyDataset**: Inherits from `TensorDataset`. This class handles the loading and preprocessing of the CSV data files. It includes methods for splitting the data into training, validation, and test sets, cleaning the data (handling missing values and scaling), and converting data to PyTorch tensors.

2. **TennyPredictionDataset**: Inherits from `TennyDataset`. This class is used for preparing the dataset specifically for prediction purposes.

3. **Tenny**: Inherits from `nn.Module`, indicating that it's a neural network model. The architecture consists of linear layers (`nn.Linear`) and dropout layers for regularization (`AlphaDropout`). The model uses ReLU activations. The `forward` method defines the forward pass through the network.

#### Notes on AlphaDropout

`AlphaDropout` is a variant of dropout, a regularization technique used in neural networks, designed specifically for models using the SELU (Scaled Exponential Linear Unit) activation function. Its purpose is to maintain the properties of the inputs (mean and variance) for which SELU is normalized during training. We considered this dropout variant in the Tenny regression model as an alternative to standard dropout just for the sake of experimentation and illustration purposes. Here's a more detailed explanation of `AlphaDropout`. 

1. **Compatibility with Activation Function**: 
   - AlphaDropout is most effective when used with SELU activation functions. If Tenny uses ReLU or other non-SELU activations, the benefits of AlphaDropout diminish, as it's tailored to preserve the mean and variance properties that SELU establishes.
   - If Tenny uses SELU, then AlphaDropout could be appropriate. However, if ReLU or another activation function is used, standard dropout or no dropout might be more suitable.

2. **Impact on Training**:
   - AlphaDropout randomly sets some input units to zero during training, helping to prevent overfitting by reducing the model's reliance on any single node. 
   - For regression models, which often deal with continuous data and sometimes subtle relationships, too much dropout (including AlphaDropout) can disrupt these delicate patterns, potentially leading to underfitting.

3. **Model Complexity**:
   - The necessity of dropout, including AlphaDropout, depends on the complexity of the model. In a relatively simple model, dropout might be unnecessary or even detrimental, as there's less risk of overfitting. 
   - In a complex model with many layers and neurons, dropout can be beneficial to prevent overfitting, but the choice between AlphaDropout and standard dropout depends on the chosen activation function.

4. **Data Characteristics**:
   - The effectiveness of any regularization technique, including AlphaDropout, also depends on the characteristics of the data. For example, if the data is not too complex or noisy, heavy regularization might not be needed.

Whether AlphaDropout is an overkill or not for the Tenny regression model depends on factors like the model's architecture (particularly the activation function), the complexity of the model, the nature of the data, and the specific problem being addressed. If the model does not use SELU or if the data and model are not complex enough to warrant such strong regularization, then AlphaDropout might not be the most suitable choice.

### `tenny-the-analyst-v4-torch.py`: The Main Codebase

The main code for classification resides in the `tenny-the-analyst-v4-torch.py` file. This file encompasses various classes.

#### TennyClassifierDataset

This class inherits from `TennyDataset` and tailors certain methods to suit the classification task. A key method in this class is `prepare_features_labels`, which performs the crucial task of extracting features and labels from the dataset. It also converts the labels from their original format to a format suitable for classification. This method is vital for ensuring the data is correctly prepared for training the model. If there are any issues during training, investigating this method is a sensible starting point.

An important aspect to note is the handling of label encoding. The `one_hot_encode` static method is utilized for one-hot encoding labels. However, PyTorch requires labels in a specific format for classification tasks, particularly as class indices (0, 1, or 2) rather than one-hot encoded vectors. The `prepare_features_labels` method takes care of converting the one-hot encoded labels to class indices.

When making predictions, the process is reversed: the class indices are converted back to the original categorical labels. This conversion is critical for interpreting the model's output in a meaningful way.

```python
CLASS_INDICES = {GROWTH_STOCK: 0, STALWART_STOCK: 1, OTHER_STOCK: 2}
CLASS_LABELS = {0: 'Growth', 1: 'Stalwart', 2: 'Other'}
...
def one_hot_encode(stock_category):
    return CLASS_INDICES[stock_category]
...

def predict_single(model, prediction_dataset):
...
    return CLASS_LABELS[predicted_index]
```

### TennyClassifierPredictionDataset

This class is tailored for making predictions with the TennyClassifier model. The `TennyClassifierPredictionDataset` inherits from `TennyPredictionDataset`, leveraging the classification-specific data processing capabilities developed in `TennyClassifierDataset`, including the handling of label encoding and feature management. 

A key aspect of `TennyClassifierPredictionDataset` is its focus on preparing data specifically for prediction. It's crucial to process the data such that predictions are made based on the most recent 'Label', as prior labels are not relevant for future predictions. 

The `read_and_clean_data` method overrides its parent class's method to add an aggregation step, ensuring the dataset is suitable for the prediction context. The `aggregate_data` method, unique to this class, processes the features, particularly constant features like 'Beta', 'Industry Beta Average', and 'Industry PE Average', using only their most recent values.

```python
class TennyClassifierPredictionDataset(TennyPredictionDataset):

    def read_and_clean_data(self, files):
        # Inherits and extends the data reading and cleaning process
        data = super().read_and_clean_data(files)
        aggregated_data = self.aggregate_data(data)
        return aggregated_data

    def aggregate_data(self, data_df):
        # Aggregates the data, particularly focusing on constant features
        aggregated_features = []
        for column in data_df.columns:
            if column in ['Beta', 'Industry Beta Average', 'Industry PE Average']:
                # For constant features, only the most recent value is used
                aggregated_features.append(data_df[column].iloc[-1])
            else:
                # Other features are extended as they are
                aggregated_features.extend(data_df[column])
        return np.array(aggregated_features)

    def __getitem__(self, idx):
        # Returns a tuple of features and labels for the given index
        return self.features[idx], self.labels[idx]
```

This class plays a crucial role in ensuring that the data fed into the TennyClassifier for predictions is processed and aggregated correctly, focusing on the most recent and relevant data points for accurate forecasting.

### TennyClassifier

The `TennyClassifier` inherits from the base `Tenny` model, reusing its architectural framework while adapting it for a multi-class classification task. This adaptation is crucial to transform the originally regression-oriented `Tenny` model into a classifier capable of handling the complexities of stock classification.

#### Key Aspects:
- **Model Balance**: Achieving the right balance between simplicity and complexity is essential. The `TennyClassifier` model aims to strike this balance, being neither overly simplistic nor excessively complex.
- **Model Configuration**: It is tailored based on the problem at hand, with no one-size-fits-all solution. The model's complexity is adjusted through experimentation and informed by experience.

#### Variants:
- **Complex Version (`TennyClassifier`)**: Designed for scenarios requiring the capture of intricate data patterns. It includes multiple layers and neurons, making it suitable for complex, non-linear financial data.
- **Simple Version (`TennyClassifierSimple`)**: A more streamlined variant, focusing on faster training and better interpretability, suitable for less complex datasets.

#### Pros and Cons:
- **Complex Model (`TennyClassifier`)**:
  - **Pros**: Better at learning complex patterns, more flexible, and performs well with large datasets.
  - **Cons**: Higher risk of overfitting, requires more data and computational power, and is less interpretable.
- **Simple Model (`TennyClassifierSimple`)**:
  - **Pros**: Less prone to overfitting, faster training and prediction, and easier to interpret.
  - **Cons**: May not capture all data complexities, potentially underfits with large datasets, and less flexible.

The choice between `TennyClassifier` and `TennyClassifierSimple` depends on the data characteristics and the specific requirements of the stock classification task. It's recommended to experiment with both models to determine which one yields better performance for your specific dataset, considering the trade-offs between complexity and interpretability.

You can set `USE_COMPLEX_MODEL` to True or False to switch between the two models:

```python
# Model to Use: Complex vs. Simple
USE_COMPLEX_MODEL = False
```

We have two versions of the `TennyClassifier` model, one complex and one simple. 

```python
class TennyClassifier(Tenny):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TennyClassifier, self).__init__(input_size, hidden_size, num_classes)
        self.fc5 = nn.Linear(hidden_size, num_classes)  # Adjusting the output layer for num_classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)  # Applying softmax to get probabilities


class TennyClassifierSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TennyClassifierSimple, self).__init__()

        # Adjust the number of layers and neurons per layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Reduced number of neurons
        self.dropout = nn.Dropout(0.3)  # Adjusted dropout rate
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # Directly connecting to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # Applying softmax to get probabilities

```

### Training and Prediction

To adapt our existing code for a classification task with the `TennyClassifier`, there are several key adjustments to consider for training, testing, and prediction.

#### Loss Function: Mean Squared Error (MSE) Loss vs. Cross-Entropy Loss

Since we're now dealing with a classification problem, the loss function should be changed from Mean Squared Error (MSE) to a classification-appropriate one like Cross-Entropy Loss.

In machine learning, different types of problems require different loss functions to guide the training of models. Two common loss functions are Mean Squared Error (MSE) and Cross-Entropy Loss. Understanding their differences is crucial, as each is suited to different types of tasks.

##### Mean Squared Error (MSE) Loss

![mse.png](images%2Fmse.png)

MSE essentially gauges the discrepancy between actual and predicted values. It's an effective metric for regression problems where predicting continuous values is the objective.


1. **Usage**: Primarily used in regression problems, where the goal is to predict continuous values. We used this one in our previous Tenny model since it was designed for predicting stock prices: a regression task.

2. **Calculation**:
   - It measures the average squared difference between the actual and predicted values.
   - Where `y_i` is the actual value and `y_i_hat` is the predicted value for the `i-th` data point, and `n` is the number of data points.

3. **Characteristics**:
   - MSE penalizes larger errors more than smaller ones due to the squaring part of the formula.
   - It’s sensitive to outliers in the data.

4. **Goal**: Minimizing MSE leads to a model that aims to predict as close to the actual values as possible.

#### Cross-Entropy Loss

![cross-entropy.png](images%2Fcross-entropy.png)

Understanding the mathematics behind Cross-Entropy Loss isn't essential. The formulas provided here are just for reference. The main point to grasp is that Cross-Entropy Loss quantifies the disparity between two probability distributions: the actual label distribution and the predicted label distribution. In essence, while MSE assesses the difference between actual numerical values, Cross-Entropy evaluates the variance in the probability distribution of outcomes.

Many experienced programmers don't fully comprehend the mathematics underlying the code they write. These mathematical components are often supplied by the frameworks or libraries in use. However, it's crucial to have a thorough understanding of the concepts and principles behind your code. This understanding is what distinguishes you as a skilled programmer.Without this understanding, many people end up using MSE and Cross-Entropy Loss interchangeably, not fully grasping the differences between the two. Ultimately, this leads to a flawed model and a lack of understanding as to why it's not performing as expected.

1. **Usage**: Used in classification problems, especially when predicting probabilities for categorical outcomes.

2. **Calculation**:
   - It measures the difference between two probability distributions: the actual label distribution and the predicted label distribution.
   - Shown here is the formula for Binary Cross-Entropy where ₩ is the actual label (0 or 1), and `p_i_hat` is the predicted probability for the `i-th` data point.

3. **Characteristics**:
   - It’s effective when the model's output represents the probability of an outcome.
   - Cross-entropy loss provides a measure of how different the predicted probability distribution is from the actual distribution.

4. **Goal**: Minimizing cross-entropy loss leads to a model that aims for a predicted probability distribution as close to the actual distribution of the labels as possible.

#### Key Differences:

- **Problem Type**: MSE is suited for regression (continuous output), while Cross-Entropy is used for classification (categorical output).
- **Output Interpretation**: MSE measures the difference in actual numerical values, while Cross-Entropy measures the difference in the probability distribution of outcomes.
- **Behavior and Sensitivity**: MSE is more sensitive to outliers than Cross-Entropy. Cross-Entropy can lead to faster convergence on classification problems, especially when using softmax in the output layer.

Choosing the right loss function is crucial in machine learning as it directly influences how the model learns during training. MSE is best when the task is to predict a continuous quantity and Cross-Entropy when predicting probabilities for categorical outcomes.

I hope you're recognizing the pattern in our entire journey. Yes, the foundation of all this AI work lies in basic math and statistics. Don't overlook these essentials if you aim to excel as an AI engineer. Skipping them is setting yourself up for failure right from the start.

### Getting the Most Out Of Small Datasets

We're working with a quite small dataset: just 12 tech companies. Therefore, we'll use as many as possible for training and testing purposes. The plan is to use 11 companies for training and one for predictions, cycling through the given tickers.

```python
TICKERS = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']
```

For example, we might begin by setting 'AAPL' aside for prediction, training the model with the other 11 companies. After predicting for 'AAPL', we move to 'MSFT', train the model anew, and continue this cycle until we've predicted for each of the 12 companies.

This technique is quite clever, isn't it? The crucial part is ensuring that Tenny doesn't get any unfair advantages. Tenny must make predictions using unseen data each time; otherwise, it wouldn't be a valid test of its predictive skills.

This approach may not be viable in all situations, but it's an effective way to assess your model when dealing with limited data. Plus, it fits our current need for personal use quite well.

To implement a simple loop in the `main` section of our code where Tenny is trained on data from 11 companies and then used to predict the classification for the remaining one, we can follow this approach:

1. Loop through each ticker in `TICKERS`.
2. For each iteration, use data from 11 companies for training and validation, and the data from the remaining one company for prediction.
3. Print out the ticker being used for prediction in each iteration to keep track of the progress.

```python
if __name__ == '__main__':
    scaler = StandardScaler()

    predictions = []

    for excluded_ticker in TICKERS:
        print(f"Training model on all companies except {excluded_ticker}. Predicting for {excluded_ticker}.")

        # Construct the filename for the excluded ticker
        prediction_file_name = f"{FOLDER_PATH}/{CSV_FILE_PREFIX}-{excluded_ticker.lower()}.csv"

        # Filter out the file for the excluded ticker
        train_val_files = [f for f in os.listdir(FOLDER_PATH) if f != prediction_file_name]

        # Determine the label position based on the number of columns
        num_columns = len(pd.read_csv(prediction_file_name).transpose().columns)
        label_position = num_columns
        # Create train, validation, and prediction datasets
        train_dataset, val_dataset, test_dataset = TennyClassifierDataset.create_datasets(
            FOLDER_PATH, label_position=label_position, device=DEVICE, scaler=scaler,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, fit_scaler=True)
        prediction_dataset = TennyClassifierPredictionDataset(
            file_path=prediction_file_name,
            label_position=label_position, device=DEVICE, scaler=scaler, fit_scaler=False
        )

        # Define the model
        input_size = train_dataset.features.shape[1]

        # experiment with different models
        if USE_COMPLEX_MODEL:
            model = TennyClassifier(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)
        else:
            model = TennyClassifierSimple(input_size=input_size, hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES)

        criterion = nn.CrossEntropyLoss()

        # PyTorch's optimizers, such as `Adam`, have built-in support for L2 regularization via the `weight_decay` parameter.
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)

        train(model, train_dataset, val_dataset, criterion, optimizer)
        test(model, test_dataset, criterion)
        # Make predictions for the excluded ticker
        # predict(model, prediction_dataset)
        prediction = predict_single(model, prediction_dataset)

        predictions.append(f"Predictions for {excluded_ticker}: {prediction}")


print('\n'.join(predictions))
```

- In each iteration, data corresponding to one ticker (`excluded_ticker`) is set aside for prediction, and the model is trained on the rest.
- Ensure that `TennyClassifierPredictionDataset` is appropriately set up to handle the single prediction file.
- The `train`, `test`, and `predict` functions should be adapted to handle the datasets and model for classification.
- This loop allows for a comprehensive training and validation cycle, ensuring that the model is tested on each company's data.

By following this approach, we can train and evaluate Tenny's performance across a variety of datasets, enhancing its robustness and generalization capabilities.

### L2 Regularization - A Streamlined Approach in PyTorch

L2 regularization is a common technique used to prevent overfitting by penalizing large weights in a neural network model. While we manually implemented L2 regularization in the `train` function of the regression model Tenny, PyTorch offers a more streamlined approach.

In the previous implementation, we added L2 regularization by iterating over model parameters:

```python
# Previous approach for L2 regularization
L2_LAMBDA = 0.001
...
l2_reg = torch.tensor(0.).to(DEVICE)
for param in model.parameters():
    l2_reg += torch.norm(param, 2)
loss += L2_LAMBDA * l2_reg
```

However, PyTorch's optimizers, such as `Adam`, provide built-in support for L2 regularization through the `weight_decay` parameter, making the manual approach unnecessary and less efficient.

#### Using the `weight_decay` Argument

We can leverage the `weight_decay` parameter in the optimizer setup to incorporate L2 regularization efficiently:

```python
# Setting up the optimizer with L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
```

With this setup, the optimizer internally manages L2 regularization, removing the need for explicit code in the training loop. This not only simplifies the code but also aligns with best practices in PyTorch. Here's the revised `train` function:

```python
# Simplified training function with L2 regularization handled by the optimizer
def train(model, train_dataset, val_dataset, criterion, optimizer):
    ...
    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    ...
```

In this updated version, L1 and L2 regularization code is omitted as `weight_decay` in the optimizer takes care of L2 regularization.

#### Key Advantages

- **Efficiency**: The optimizer updates model parameters considering the regularization term, thus streamlining the training process.
- **Canonical Approach**: Using `weight_decay` is a widely accepted method in PyTorch and adheres to best practices.
- **Adaptation of L2 Regularization**: The `weight_decay` method in PyTorch slightly differs from classical L2 regularization by not using the learning rate as a coefficient, but it effectively serves the same purpose of penalizing large weights.

By integrating L2 regularization through the `weight_decay` argument, we ensure a more efficient and standardized implementation in PyTorch, contributing to a cleaner and more maintainable codebase.

### Scaling and Normalization Revisited

Feature scaling is a crucial preprocessing step in machine learning, particularly when working with data that varies in scale across different features. This process standardizes the range of independent variables or features within the data. In simpler terms, it ensures all input features (like height, weight, temperature, etc.) are on a similar scale. This standardization is important because many machine learning algorithms perform better or converge faster when features are scaled similarly and are close to a normal distribution.

#### Utilizing `StandardScaler` from `scikit-learn`

In our code, we employ `StandardScaler` from the `scikit-learn` library for this purpose. `StandardScaler` standardizes features by removing the mean and scaling to unit variance. This technique is a form of normalization (or standardization) and is crucial for preparing the data for machine learning models.

#### Process in Our `create_datasets` Method

1. **Initialize the Scaler:**
   - The `scaler` is an instance of `StandardScaler`, prepared before dataset creation.
   - **Fit:** Calculates the mean and standard deviation for each feature in the training dataset.
   - **Transform:** Applies the scaling parameters to the data.

2. **Creating Datasets:**
   - The `create_datasets` method generates training, validation, and test datasets.
   - It ensures that data is appropriately scaled for the model.

   ```python
    @staticmethod
    def create_datasets(folder_path, label_position, device, scaler, train_ratio, val_ratio, fit_scaler=False):
        # Create the train dataset
        train_dataset = TennyDataset(folder_path, label_position, device, scaler, fit_scaler)

        # Create the validation and test datasets
        val_dataset = TennyDataset(folder_path, label_position, device, scaler=scaler)
        test_dataset = TennyDataset(folder_path, label_position, device, scaler=scaler)

        return train_dataset, val_dataset, test_dataset
   ```

3. **Fitting Scaler to Training Data:**
   - The scaler is fitted to the training data if `fit_scaler=True` is set. This computes the mean and standard deviation used for scaling.

4. **Transforming Non-Training Data:**
   - Validation and test datasets are scaled using the same parameters derived from the training data, without refitting the scaler.

5. **Ensuring Proper Scaling:**
   - The `__init__` method of `TennyDataset` applies the scaler's `transform` method to scale the features.

```python
class TennyDataset(TensorDataset):
    def __init__(self, folder_path, label_position, device='cpu', scaler=None, fit_scaler=False):
...
        if fit_scaler:
            scaler.fit(self.features)

...
```

#### Key Considerations

- **Uniform Scaling**: All data, including training, validation, and test sets, should be scaled using the same parameters to ensure consistency.
- **No Refitting on Prediction Data**: When creating prediction datasets, the scaler is not refitted. Instead, it applies the scaling parameters from the training data.

#### Data Format and Normalization

The data format in our dataset consists of various financial metrics with different ranges. The `StandardScaler` plays a critical role in normalizing these features to a uniform scale, enhancing the model's ability to learn from the data effectively.

### Output Size - Number of Classes

The output size in a classification model like `TennyClassifier` is a crucial component that must correspond to the number of classes in the classification task. This determines the dimensionality of the output layer of the model, which, in turn, influences the model's ability to differentiate between the various classes.

For example, in our case where we have three classes ('Growth', 'Stalwart', 'Other'), the output size of the model should be 3. This ensures that the model's final layer produces outputs that can be mapped to these three distinct classes.

#### Implementing Flexible Output Size

To maintain flexibility and good coding practice, it's recommended to avoid hard-coding the number of classes directly into the model. Instead, the number of classes should be passed as a parameter to the model's constructor (`__init__`). This approach allows the model to be more adaptable to different datasets with varying numbers of classes.

```python
# Define the number of classes as a constant
NUM_CLASSES = 3  # Growth, Stalwart, Other

# TennyClassifier definition with dynamic output size
class TennyClassifier(Tenny):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TennyClassifier, self).__init__(input_size, hidden_size, num_classes)
        ...
        # The final fully connected layer's output size is set based on `num_classes`
        self.fc5 = nn.Linear(hidden_size, num_classes)  # Adjusting the output layer for num_classes
        ...

class TennyClassifierSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TennyClassifierSimple, self).__init__()
        ...
        # The final fully connected layer's output size is set based on `num_classes`
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # Directly connecting to output layer
...
```

In this configuration, the `num_classes` parameter in the `TennyClassifier` constructor allows for the dynamic setting of the output size. This design not only aligns with best practices in software development but also enhances the model's versatility, making it suitable for various classification tasks with different numbers of classes.

### Evaluation Metrics

For classification tasks, monitoring metrics such as accuracy, precision, recall, and F1-score is crucial. These metrics provide a more detailed understanding of the model's performance compared to solely relying on the loss value.

#### Incorporating Evaluation Metrics into Training and Testing

In the `train` and `test` functions, we'll calculate and display these additional metrics to gain a better insight into the model's performance:

1. **Accuracy**: Measures the proportion of correct predictions.
2. **Precision and Recall**: Precision measures the accuracy of positive predictions, while recall (sensitivity) measures the ability of the model to find all the positive instances.
3. **F1-Score**: A weighted average of precision and recall.

Here's we modify our `train` and `test` functions to include these metrics:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics(outputs, labels):
    # Convert outputs to predicted classes
    _, predicted_classes = torch.max(outputs, 1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels.cpu(), predicted_classes.cpu())
    precision = precision_score(labels.cpu(), predicted_classes.cpu(), average='macro')
    recall = recall_score(labels.cpu(), predicted_classes.cpu(), average='macro')
    f1 = f1_score(labels.cpu(), predicted_classes.cpu(), average='macro')
    
    return accuracy, precision, recall, f1

def train(model, train_dataset, val_dataset, criterion, optimizer):
    ...
    for epoch in range(NUM_EPOCHS):
        ...
        # Training phase
        ...
        # Validation phase
        model.eval()
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Transfer data to the device
                outputs = model(inputs)  # Forward pass: compute predicted outputs by passing inputs to the model
                val_loss += criterion(outputs, labels).item()
                acc, prec, rec, f1 = evaluate_metrics(outputs, labels)
                val_accuracy += acc
                val_precision += prec
                val_recall += rec
                val_f1 += f1
        # Average the metrics over the validation set
        val_accuracy /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Prec: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
        ...
```

For the `test` function:

```python
def test(model, test_dataset, criterion):
    ...
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            ...
            test_loss += criterion(outputs, labels).item()
            acc, prec, rec, f1 = evaluate_metrics(outputs, labels)
            test_accuracy += acc
            test_precision += prec
            test_recall += rec
            test_f1 += f1
    # Average the metrics over the test set
    test_accuracy /= len(test_loader)
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_f1 /= len(test_loader)
    test_loss /= len(test_loader)

    print(f"Average Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test Prec: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
```
### Understanding Undefined Metric Warnings in Classification Tasks

In classification tasks, particularly when evaluating a model's performance, it's common to encounter warnings related to undefined metrics. These warnings typically arise in the context of calculating precision and recall, two key performance metrics.

```python
UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
```

#### Precision and Recall: A Quick Recap

- **Precision**: Precision measures how many of the items identified as positive by the model are actually positive. It's a ratio of true positives (correct positive predictions) to all positives predicted by the model (both true positives and false positives).
- **Recall**: Recall measures how many of the actual positive items are correctly identified by the model. It's a ratio of true positives to the sum of true positives and false negatives (the positives the model failed to identify).

#### The Scenario Leading to Undefined Metrics

1. **Precision Warning**:
   - **Cause**: This warning appears when your model does not make any positive predictions for a specific class – in other words, both true positives (TP) and false positives (FP) are zero for that class. Since precision is calculated as TP / (TP + FP), this scenario leads to a division by zero, making precision undefined.
   - **Example**: Imagine a scenario where your model is supposed to identify apples among various fruits, but it never labels any fruit as an apple. In this case, precision for the 'apple' class becomes undefined.

2. **Recall Warning**:
   - **Cause**: This warning occurs when there are no true positives and false negatives for a specific class. This happens when the class is not present in the actual labels of your test set, making recall, calculated as TP / (TP + FN), undefined.
   - **Example**: Consider a situation where you're testing your model to recognize various animals, but there are no cats in your test dataset. The recall for the 'cat' class becomes undefined since the model never had a chance to correctly or incorrectly identify a cat.

#### Addressing the Warnings

- **Zero Division Parameter**: In both cases, we can control the behavior when faced with division by zero using the `zero_division` parameter in functions like `precision_score` and `recall_score`. Setting `zero_division=1` assumes a perfect score in these undefined scenarios, while `zero_division=0` assigns a score of zero.
- **Data and Model Review**: Frequently encountering these warnings can indicate issues with your model or the way your data is distributed. It might suggest that your model is struggling to identify certain classes, or there might be an imbalance in your dataset.

The issue we're facing with the `UndefinedMetricWarning` for precision and recall is likely due to the highly imbalanced nature of our dataset. When you have a dataset with a significant imbalance in the class distribution – in our case, having only one 'Growth' and one 'Other', with the rest being 'Stalwart' – it can lead to several challenges.

#### Challenges of Imbalanced Datasets

1. **Model Bias Towards Majority Class**: 
   - The model tends to predict the majority class ('Stalwart' in our case) more often because it has been exposed to it more during training. This can result in very few or no predictions for the minority classes ('Growth' and 'Other').

2. **Difficulty in Learning Minority Classes**: 
   - The model has limited exposure to the minority classes, making it difficult to learn their characteristics. As a result, it might fail to identify these classes correctly.

3. **Metrics Becoming Misleading or Undefined**: 
   - Standard metrics like accuracy can be misleading, as a model biased towards the majority class might still achieve high accuracy.
   - Precision and recall for minority classes can become undefined if the model fails to make any true positive predictions for those classes.

#### Strategies to Address Imbalance

1. **Resampling Techniques**: 
   - **Oversampling Minority Classes**: Increase the number of samples from the minority classes, possibly using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic samples.
   - **Undersampling Majority Class**: Reduce the number of samples from the majority class to balance the class distribution.

2. **Altering Model Thresholds**: 
   - Adjust the decision threshold of the classifier to be more sensitive to the minority classes.

3. **Using Different Metrics**: 
   - Focus on metrics that handle imbalance better, such as the F1-score, Matthews correlation coefficient, or area under the ROC curve (AUC-ROC).

4. **Weighted Loss Function**: 
   - Use a weighted loss function in training that gives more importance to the minority classes.

5. **Ensemble Techniques**: 
   - Use ensemble methods that can combine multiple weak learners to improve the model's performance on imbalanced data.

### Implementing Solutions

Given our specific scenario, here are some steps we can take:

- **Resample our Data**: Implement oversampling for 'Growth' and 'Other' classes. Since we are using PyTorch, we can do this when creating your DataLoader by using the `WeightedRandomSampler` to give more weight to minority classes.

- **Modify Evaluation Metrics**: Use metrics that are more informative for imbalanced datasets, and consider setting `zero_division=1` in precision and recall calculations to avoid undefined metrics.

- **Experiment with Different Models**: Some models might be more robust to class imbalance. Experimenting with different architectures or algorithms could yield better results.

Addressing the class imbalance in your dataset will help improve the model's ability to learn from the minority classes and give a more accurate picture of its performance.

Let's resample our data using `WeightedRandomSampler` to give more weight to minority classes. Implementing oversampling in PyTorch using `WeightedRandomSampler` is an effective way to address class imbalance in your dataset. Here's how we can do it:

#### Step-by-Step Guide to Implementing Oversampling with `WeightedRandomSampler`

1. **Calculate Weights for Each Class**:
   - First, determine the weight for each class. The idea is to assign a higher weight to minority classes so that they are sampled more frequently during training.
   - One common approach is to use the inverse of class frequencies.

2. **Assign Weights to Each Sample**:
   - Next, assign each sample in the dataset a weight based on its class.

3. **Create a `WeightedRandomSampler`**:
   - Use these weights to create an instance of `WeightedRandomSampler`.

4. **Use the Sampler in DataLoader**:
   - Finally, pass the `WeightedRandomSampler` to the `DataLoader`.

Here's a code example to illustrate this process:

```python
from torch.utils.data import DataLoader, WeightedRandomSampler
...
def train(model, train_dataset, val_dataset, criterion, optimizer):
    # Create weighted sampler for the training dataset
    labels = train_dataset.labels

    # Calculate the class weights
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # Create the DataLoader with the sampler
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

```

### Points to Note

- **DataLoader Shuffle**: When using a sampler, you should not set `shuffle=True` in your `DataLoader`, as the sampler already takes care of the shuffling based on the weights.
- **Length of Sampler**: The `num_samples` argument in `WeightedRandomSampler` can be set to the length of your dataset. If you want each epoch to see a different sample distribution, you can set `replacement=True`.
- **Evaluation**: While oversampling can improve performance on the minority class, it's important to evaluate the model on a non-oversampled validation set to understand its true performance.

By implementing oversampling in this way, we increase the chances of our model seeing and learning from the minority class examples, which can help alleviate the bias towards the majority class.

### Predicting Class Labels

At long last, we're ready to make predictions with our model.

Since we're dealing with classification, the output will be class probabilities. We may want to convert these probabilities to class labels.  

```python
def predict_single(model, prediction_dataset):
    model.eval()
    with torch.no_grad():
        # Assuming the dataset returns a tuple (features, label)
        # and we are using the first item in the dataset for prediction
        features, _ = prediction_dataset[0]
        features_tensor = features.unsqueeze(0).to(DEVICE)  # Add batch dimension and send to device

        # Get prediction from model
        prediction = model(features_tensor)

        # Process the prediction
        predicted_index = torch.argmax(prediction, dim=1).item()
        return CLASS_LABELS[predicted_index]

```

1. **Data Preparation**:
   - The function correctly extracts features for prediction and adds a batch dimension. This is important because PyTorch models expect inputs in a batch format, even if the batch size is 1.

2. **Model Evaluation Mode**:
   - The use of `model.eval()` is appropriate as it sets the model to evaluation mode, affecting layers like dropout and batch normalization, which behave differently during training and inference.

3. **No Gradient Calculations**:
   - `torch.no_grad()` is used properly to ensure that no gradient calculations are made during prediction, which saves memory and computation.

4. **Predicted Class Extraction**:
   - The prediction is processed to extract the class index with the highest probability (`torch.argmax`) and then mapped to the corresponding class label (`CLASS_LABELS` dictionary). This is the standard approach for classification tasks.

### Final Notes on Training and Prediction

In our approach, we intentionally re-initialize the model in each iteration of the main loop. While this is not a standard practice in typical machine learning workflows, it is deliberately chosen for our specific goal of training and predicting for each of the 12 companies separately. This method allows us to assess the model's performance individually across different datasets.

1. **Model Saving for Efficiency**:
   - If training proves to be time-consuming, consider saving the model post-training (`torch.save(model.state_dict(), 'model.pth')`) and loading it for prediction (`model.load_state_dict(torch.load('model.pth'))`). This practice can significantly reduce the time and computational resources required by eliminating the need for retraining the model each time.

2. **Consistency in Data Processing**:
   - Ensure that the `prediction_dataset` is properly formatted and scaled. Consistent and accurate data processing is essential, especially in workflows with varied data transformations.
   - The scaler should be fitted on the training data to learn its distribution and then used to scale the validation, test, and prediction datasets, ensuring uniformity across all data used in the model.

3. **Device Consistency**:
   - Maintain consistent use of the computing device (`DEVICE`) throughout the training, evaluation, and prediction phases to ensure seamless model operation.

### Exploring Hyperparameters in "Tenny, the Classifier"

In our journey with "Tenny, the Classifier," particularly given our dataset's modest size, playing with hyperparameters becomes a crucial aspect of model optimization. Hyperparameters are adjustable parameters that let you control the model training process and can significantly impact the performance of the model. Let's delve into the hyperparameters we've set and consider potential adjustments:

#### Key Hyperparameters

1. **NUM_EPOCHS (1000)**:
   - **Purpose**: Represents the number of times the entire dataset is passed forward and backward through the neural network.
   - **Consideration**: With a small dataset, 1000 epochs might be excessive and could lead to overfitting. Monitor the training process and consider reducing this number if early stopping frequently kicks in.

2. **BATCH_SIZE (20)**:
   - **Purpose**: Determines the number of samples processed before the model is updated.
   - **Consideration**: A batch size of 20 is moderate, balancing computational efficiency with the benefits of stochastic gradient descent. Experiment with smaller or larger sizes to observe the impact on training dynamics and model performance.

3. **HIDDEN_SIZE (20)**:
   - **Purpose**: The size of the hidden layers in the neural network.
   - **Consideration**: This should be tuned according to the complexity of the dataset. For a small dataset, a smaller hidden size might suffice, reducing the risk of overfitting.

4. **LEARNING_RATE (0.0001)**:
   - **Purpose**: Controls how much to change the model in response to the estimated error each time the model weights are updated.
   - **Consideration**: A lower learning rate ensures more precise adjustments to weights, though it makes the training slower. Experiment with varying this rate to find a balance between speed and accuracy.

5. **TRAIN_RATIO (0.7) and VAL_RATIO (0.2)**:
   - **Purpose**: These ratios define how the dataset is split into training, validation, and testing sets.
   - **Consideration**: These ratios seem reasonable, but you might experiment with different splits, especially if you observe overfitting or underfitting. Keep in mind that the ratios specified here should correspond with those established in the `__init__.py` file within the `tenny` package.

6. **PATIENCE (20)**:
   - **Purpose**: Used for early stopping, this parameter defines the number of epochs to wait for improvement in validation loss before stopping training.
   - **Consideration**: Adjust based on how quickly your model converges. Too small a value might stop training prematurely, while too large a value could lead to wasted computational resources.

7. **L1_LAMBDA and L2_LAMBDA (0.001)**:
   - **Purpose**: These are regularization parameters to prevent overfitting.
   - **Consideration**: The effectiveness of these values depends on the model's complexity and dataset size. Adjust them if you observe overfitting. `L1_LAMBDA` is not used in the current implementation.

8. **USE_COMPLEX_MODEL (True)**:
   - **Purpose**: Determines whether to use a more complex or simpler model architecture.
   - **Consideration**: Given the small dataset, a simpler model might be sufficient and reduce the risk of overfitting. Consider toggling this parameter to compare performances.

#### Approach to Hyperparameter Tuning

- **Gradual Adjustment**: Avoid changing all hyperparameters at once. Instead, adjust one or two at a time and observe the impact.
- **Monitor Performance**: Regularly monitor the training and validation performance to identify signs of overfitting, underfitting, or other issues.
- **Experimentation**: Don't hesitate to experiment with significantly different values to see their impact.

In essence, fine-tuning these hyperparameters can significantly enhance the performance of "Tenny, the Classifier." Given the dataset's size, a more cautious approach with smaller, simpler models and moderate training iterations could be beneficial. As always, the goal is to find the sweet spot where the model generalizes well without overfitting or underfitting the data.

### Don't Expect AIs To Mirror Your Opinions

This code serves as an educational tool, incorporating best practices for learning purposes. It is not designed for production use but rather to illustrate various aspects of a machine learning project. While the code includes many best practices, some features might be more elaborate than required for practical applications. It's crucial to adapt and scale solutions based on the specific needs and constraints of your project.

Remember, AI models are not designed to merely echo our opinions. Their value lies in offering different, and potentially more accurate, perspectives. It's essential to approach AI with an open mind, recognizing that its conclusions might differ from our initial expectations or beliefs.

[The-History-of-Human-Folly.md](..%2F..%2Fessays%2FAI%2FThe-History-of-Human-Folly.md)

I encourage you to expand the dataset by including more companies and to experiment with varying hyperparameters. Exploring different model architectures and algorithms can also be insightful to see how they affect performance. 

If you decide to add new companies, make sure their data matches the format of the current files. It's important to source this data from a reliable provider and verify its recency. Experimenting with various combinations of companies can provide deeper insights into the model's performance.

Remember, for the most accurate and reliable data, I used Kyofin, which underscores the notion that investing in quality data is often necessary.

## Conclusion of Part II: "Tenny, the Classifier"

With this, we bring Part II of our journey with "Tenny, the Classifier" to a close. Throughout this phase, we've navigated various challenges and intricacies of AI and machine learning, always with an eye toward both learning and application. 

As we continue to delve deeper into this fascinating field, let us keep in mind a thought that resonates profoundly: 

> "The Journey is the Reward."

This sentiment echoes the spirit of our endeavor. In the world of AI and machine learning, each step, each discovery, each challenge overcome is not just a means to an end but a reward in its own right. It's a reminder that the knowledge we gain and the experiences we accumulate along the way are invaluable, shaping our understanding and approach to future challenges.

May this quote serve as a source of inspiration and motivation as we forge ahead, exploring new horizons and uncovering the vast potential of artificial intelligence.
