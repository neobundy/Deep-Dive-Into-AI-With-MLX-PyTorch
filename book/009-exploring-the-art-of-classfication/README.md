# Part II

In Part II, we're set to dive into more sophisticated AI concepts and techniques. While a detailed roadmap isn't laid out yet, we'll tackle these concepts one at a time, ensuring a thorough understanding.

We'll kick things off with **Classification**. Both regression and classification are fundamental in the world of machine learning and AI, laying the groundwork for grasping more intricate AI models and applications. Having already explored regression in Part I, it's time to delve into classification and its nuances.

# Chapter 9 - Exploring the Art of Classification with Tenny

![tenny.png](images%2Ftenny.png)

In Chapter 9, we venture further into the bustling AI kitchen, moving from the structured world of regression to the nuanced realm of classification. With Tenny as our trusty sous-chef, we've adeptly tackled the task of stock price prediction, akin to carefully measuring and mixing ingredients. Now, we turn our attention to a different culinary skill: sorting and categorizing a diverse array of ingredients – a process parallel to classifying stocks into distinct groups like 'Growth', 'Stalwart', and 'Speculative(or Other)'. This chapter is about expanding our AI culinary techniques, applying Tenny's analytical prowess to discern and categorize the complex flavors of the financial market. It's a journey from following recipes to understanding the essence of ingredients, embracing the varied and often unpredictable nature of stock analysis. Let's step into this new section of our AI kitchen, ready to explore the rich and varied tastes that stock classification has to offer.

Regression and classification are indeed considered the foundational building blocks in the field of machine learning and AI. They introduce the core principles and techniques that are essential for understanding more complex AI models and applications. 

1. **Regression**: This is typically where many AI and machine learning courses start. Regression models, which predict continuous values, are fundamental for understanding how machines can interpret and infer from numerical data. Concepts learned in regression, such as loss functions, overfitting, and model evaluation, are crucial for grasping more advanced AI topics.

2. **Classification**: Following regression, classification introduces the concept of predicting discrete labels or categories. This fundamental machine learning task broadens the understanding of AI applications, especially in scenarios where the output is categorical (like spam detection, image recognition, etc.). It also introduces important concepts like decision boundaries, probability-based classifiers, and performance metrics specific to classification tasks.

Together, these two topics lay a solid foundation for understanding more complex and specialized areas of AI, such as deep learning, natural language processing, and reinforcement learning. They are essential for anyone looking to build a comprehensive understanding of AI and machine learning.

## Notes on Classifying Stocks

In this chapter, we delve into a fundamental example of classification: sorting stocks into three distinct categories – 'Growth', 'Stalwart', and 'Speculative'. This framework is inspired by the legendary stock picker Peter Lynch, who used a more extensive categorization system. For our purpose, we simplify it to these three categories. 'Speculative' is an addition, serving as a catch-all for stocks that don't align with the other categories, essentially marking them as 'none-of-the-above' or 'alternative'.

The process of providing Tenny with accurate labels for learning is a critical step in supervised learning, particularly in a task like stock categorization. As a seasoned investor undertaking this as a personal project, I will rely on my own judgment and experience to classify these stocks. This approach, while subjective, is rooted in a deep understanding of the market and years of investment experience. It's important to note that while this method reflects a personal investment philosophy, it may not be the standard approach in a more objective, production-grade system. In real-world scenarios, a broader and more varied set of expert opinions might be sought to ensure objectivity.

Here are some strategies to obtain labels for your dataset:

1. **Expert Analysis**: Collaborate with financial experts or analysts who can label the data based on their knowledge and experience. This approach can ensure a high level of accuracy but can be time-consuming and costly.

2. **Historical Performance Metrics**: Use quantifiable and objective criteria to label stocks. For instance:
   - **Growth Stocks**: Could be labeled based on historical earnings growth rate, stock price appreciation, etc.
   - **Stalwart Stocks**: Identified by stable earnings, moderate growth, and consistent dividend payouts.
   - **Speculative Stocks**: Characterized by high volatility, lower market cap, or being in emerging sectors.

3. **Rule-Based Labeling**: Develop a set of rules or criteria to automatically label the stocks. This could involve setting thresholds for various financial metrics that define each category. While this method can process large datasets efficiently, it requires careful design to ensure the rules accurately reflect each category.

4. **Publicly Available Labels**: Leverage existing datasets where stocks have been categorized by financial institutions, market research firms, or in academic studies.

5. **Crowdsourcing**: Obtain labels through crowdsourcing platforms where multiple individuals, ideally with some financial knowledge, provide labels. This method can help mitigate individual biases but requires mechanisms to ensure the quality and reliability of the labels.

6. **Semi-Supervised Learning**: If labeling an entire dataset is not feasible, you can label a small subset and use semi-supervised learning techniques to extend these labels to the larger dataset.

7. **Iterative Refinement**: Start with an initial set of labels (possibly rule-based) and iteratively refine them as Tenny is trained and evaluated. Misclassifications and model feedback can guide adjustments to the labeling criteria.

Whichever method you choose, it’s important to ensure that the labeling process is as objective and consistent as possible. The quality of Tenny's training and its eventual accuracy in classifying stocks will heavily depend on the quality and reliability of these labels.

However, let's acknowledge that at the end of the day, these are indeed opinions. Ultimately, as an investor, the final decision rests with you. This personal project is about embracing that responsibility, using your own insights and convictions to guide your classifications. The importance of doing your own due diligence cannot be overstated – your investments should be based on your own informed decisions, not solely on the advice of others.

Imagine yourself as a well-seasoned investor, armed with years of experience and a finely tuned investment strategy. You've witnessed market fluctuations, adapted your tactics, and learned valuable lessons along the way. You've developed a sharp sense for market trends and a confidence in your stock-picking abilities. This chapter is about translating that wealth of experience into a practical application with Tenny, allowing for a unique and personalized exploration of stock classification.

Given that this is a personal project of a seasoned investor, the approach to labeling and model training would indeed be more subjective and tailored to your individual investment philosophy and expertise. Here’s how you can approach this:

1. **Personal Expertise as a Basis for Labeling**: As a seasoned investor, you can use your own investment criteria and experience to label stocks. This could include your assessment of financial statements, growth potential, market trends, or any other criteria you deem relevant. Your unique perspective and expertise will provide a distinct angle to the data.

2. **Documenting Your Investment Logic**: For each stock you label, it might be helpful to document your reasoning. This not only serves as a record of your thought process but also adds depth and insight into the training data, which can be particularly valuable for readers looking to understand the rationale behind each classification.

3. **Dynamic Labeling Reflecting Market Changes**: Given your experience, you’re likely aware of how market conditions can change a stock’s category over time. You can reflect this dynamism in your labels, perhaps revisiting and adjusting them periodically to ensure they stay current with market realities.

4. **Focus on Practical Application Over Theory**: Since the aim is to avoid traditional, theoretical examples, your approach can be more practical and hands-on. Demonstrating how Tenny can be used to augment your investment strategies with real-world data will provide a unique and engaging learning experience.

5. **Incorporating Intuition and Market Sense**: In addition to financial metrics, your intuition and sense of the market, honed over years of experience, can play a significant role in how you classify stocks. This human element, often overlooked in traditional AI models, could be a unique aspect of your project.

6. **Iterative Refinement Based on Performance**: Use Tenny’s predictions and classification outcomes as a feedback loop to refine your approach. If certain predictions align or conflict with your market understanding, it could provide interesting insights for further refinement.

In essence, this approach allows Tenny to be an extension of your investment style and philosophy, offering a personalized and unique perspective on stock classification. 

### Growth vs Stalwart: Personal Investment Philosophy

I can't emphasize enough how object orientation is akin to a life hack. It's an incredibly adaptable tool, applicable across a wide range of domains, including the realm of the stock market.

Consider the concept of "Growth vs. Stalwart."

Traditionally, "Growth" and "Stalwart" are terms categorizing stocks based on their performance patterns and characteristics. These categories draw inspiration from Peter Lynch, a legendary stock picker, who utilized an extensive categorization system. 

1. **Growth Stocks**: 
   - **Characteristics**: Growth stocks are shares in companies that are expected to grow at an above-average rate compared to other companies in the market. These companies often reinvest a significant portion of their earnings into further business expansion, research and development, or acquisitions. They are typically from sectors experiencing rapid growth, such as technology or biotech.
   - **Financial Indicators**: High earnings growth rates, often accompanied by higher price-to-earnings (P/E) ratios. They generally do not pay high dividends as profits are reinvested into the business.
   - **Investor Expectation**: Investors buy growth stocks with the expectation of capital appreciation. They are usually more interested in the company's future potential than its current profitability.
   - **Risk Profile**: These stocks tend to be more volatile and carry higher risks, as their valuation often hinges on future growth prospects, which can be uncertain.

2. **Stalwart Stocks**:
   - **Characteristics**: Stalwart stocks belong to established companies with a long history of stable earnings, often in mature industries. These companies grow steadily and can weather economic downturns relatively well.
   - **Financial Indicators**: Moderate, but consistent earnings growth rates, and more reasonable P/E ratios compared to growth stocks. They often pay regular dividends, reflecting their stable earnings.
   - **Investor Expectation**: Investors are attracted to stalwart stocks for their reliability and the steady income from dividends. These stocks may not offer dramatic growth, but they provide stability and moderate growth prospects.
   - **Risk Profile**: Generally, stalwarts are considered less risky than growth stocks. They are popular choices for conservative investors who prefer steady returns over high-risk, high-reward investments.

The primary distinction between growth and stalwart stocks lies in their growth rates, risk profiles, and reinvestment strategies. Growth stocks are known for rapid expansion and high potential returns, albeit with greater risk. Stalwarts, conversely, are synonymous with steady, reliable growth and income.

Yet these are broad definitions. 'Their' definitions. 

What about 'your' definitions? What about 'your' personal criteria? How do you differentiate between growth and stalwart stocks? Which specific traits and financial metrics do you prioritize? What are your expectations and acceptable risk levels for these stocks?

The stock market is a dynamic, constantly evolving entity, a complex network of companies, investors, and market forces. As an investor, you navigate this complexity, leveraging your experience and expertise to make informed choices.

Indeed, informed decisions are essential. Your choices are not based on mere instinct or guesswork; they are the result of your market knowledge, hands-on experience, and investment philosophy.

So, the pivotal question is:

> How well-informed are your investment decisions?

Investing in stocks inherently involves risk. However, this doesn't necessitate reckless risk-taking. You can be a calculated risk-taker, well-informed about the risks, knowledgeable about the market, and adhering to a clear investment philosophy.

In this regard, growth stocks often seem more attractive than other options, speculative stocks aside. Entering the market means embracing some level of risk. Investing exclusively in stalwart stocks might appear safer, but it's not devoid of risk. If you seek a truly risk-free option, you might consider bonds, though at the expense of potentially higher returns. Therefore, the appeal of growth stocks – they promise higher returns, though with increased risk.

But remember, this is a personal perspective. I believe even if you start with supposedly super risk-free bonds, the allure of the stock market will eventually draw you in. And once involved, you'll become more accustomed to taking risks, possibly even escalating to higher-risk investments. Safer options like stalwart stocks carry their own risks and might be a mere stepping stone towards riskier choices. So why not start with growth stocks? That's my rationale, and it's why I favor growth stocks.

The philosophy here is straightforward: if you're going to take a market risk, aim for the highest return to justify that risk. Stalwart stocks might not offer this; growth stocks do.

What about higher-risk, high-return options like derivatives and cryptocurrencies?

If you're comfortable with such risks, then why not? Personally, I avoid what I consider 'dumb risks' – those contradicting calculated, long-term strategies. Don't be misled by the plethora of charts and tools promising accurate risk and gain predictions. In my view, they give a false sense of security and control, when in reality, you're just another market participant, and the potential for loss is very real. Don't be a sucker for this illusion of control.

> If you’ve been playing poker for half an hour and you still don’t know who the patsy is, you’re the patsy.

Contrary to popular belief, this quote is not from Warren Buffett but is an old poker adage. If you're merely pretending to be savvy in the stock market, you're still the patsy, the easy target.

From my standpoint, the riskiest yet potentially safest bet in the stock market is on growth stocks. That's where I place my bets.

Remember this:

> Every darn thing is an object.

This applies to stocks as well. They are dynamic entities. A stock classified as 'growth' yesterday might be 'stalwart' today and could evolve into a 'speculative' stock tomorrow. The stock market is constantly changing, and stocks adapt with it.

Stocks also demonstrate properties like inheritance and polymorphism, effectively encapsulating their unique characteristics and behaviors. It all starts with the concept of abstraction. Your task is to understand this abstraction, using the right tools – namely, the four pillars of object orientation.

Consider the shared attributes of growth stocks. You can encapsulate these commonalities into an abstract class, then create your own subclassed, concrete classes and instantiate them as individual objects.

These become _your_ growth stocks, defined by your understanding and application of object-oriented principles.

Do you see why I say object orientation is like a life cheat code? It's a powerful approach, applicable even in the stock market. 

Here's a mental model in pseudo-code:

```python
# Python code demonstrating object-oriented principles applied to stock categorization and analysis.

from abc import ABC, abstractmethod

# Abstract Class for Growth Stocks
class GrowthStock(ABC):
    def __init__(self, symbol, current_price):
        self.symbol = symbol
        self.current_price = current_price

    @abstractmethod
    def analyze_growth_potential(self):
        """Analyze and return the growth potential of the stock."""
        pass

# Concrete Subclass for a Specific Growth Stock
class TechGrowthStock(GrowthStock):
    def __init__(self, symbol, current_price, projected_growth_rate):
        super().__init__(symbol, current_price)
        self.projected_growth_rate = projected_growth_rate

    def analyze_growth_potential(self):
        # Simplified analysis based on projected growth rate
        if self.projected_growth_rate > 20:
            return "High growth potential"
        elif 10 <= self.projected_growth_rate <= 20:
            return "Moderate growth potential"
        else:
            return "Low growth potential"

# Example of instantiation
my_stock = TechGrowthStock('NO_SUCH_TICKER', 2800, 25)  # Assuming a growth rate of 25%
print(my_stock.analyze_growth_potential())  # Output: High growth potential
```
This pseudo-code illustrates how object-orientation concepts can be applied to categorize and analyze stocks, reflecting your unique investment approach.

The mental model can be as complex or simple as you need. The key is understanding the underlying principles and applying them effectively.

```python
# Simple Python code structure for categorizing stocks using object orientation.

class Stock:
    pass

class StalwartStock(Stock):
    pass    

class GrowthStock(Stock):
    pass

class TechGrowthStock(GrowthStock):
    pass
```

This is my mental model – simple yet powerful. Within this framework, you're free to creatively adapt to the dynamic nature of the stock market.

What about other life domains? Can the same principles be applied?

I posed this set of questions in the following essay:

✍️ The Normal Distribution: Using the Bell Curve as a Life Hack: https://github.com/neobundy/Deep-Dive-Into-AI-With-MLX-PyTorch/blob/master/essays/life/Normal-Distribution-As-A-Life-Hack.md

- Where do you stand in the normal distribution of 'stock investors who truly earn significant returns'?
- Where do you stand in the normal distribution of 'individuals who truly understand the stock market'?
- Where do you stand in the normal distribution of 'individuals who grasp the market, earn significant returns, and sustain them over the long term'?

They're not meant to scare you; they're meant to make you think. Some instinctively know their stuff, while others are clueless. Where do you stand? Some are born with this instinct, while others learn it. Where do you stand? 

Now, consider the above questions again. These considerations should send chills down your spine. 

The philosophy of object-oriented investing is about more than just the mechanics of investing; it's about understanding the market, your place within it, and how to leverage these principles for long-term success.

So, remember:

> Every damn thing is an object.

This mindset is not just a clever approach to investing; it’s a transformative way of thinking, applicable across various aspects of life and decision-making. By framing your understanding and actions within the object-oriented paradigm, you're not just investing in stocks; you're investing in a mindset that cultivates clarity, strategy, and adaptability – key components for success in both the stock market and life.

Now let's build on this philosophy, applying it to the task of stock classification. Remember these are my criteria, based on my investment philosophy. You can use them as a starting point, then adapt and refine them to suit your own approach.

### Common Traits of Growth Stocks

In the context of using Tenny for classifying stocks into categories like 'Growth' or 'Non-Growth', selecting the right features is crucial for the model's performance. The features should capture the various aspects of a company's financial health, stock performance, and market behavior. Here's a list of potential features that could be significant for this classification task. Again, remember this list is based on traditional growth stock characteristics, not even my personal criteria.

1. **Earnings Growth**: Measures the rate at which a company's earnings are increasing. High earnings growth is often a characteristic of growth stocks.

2. **Market Capitalization**: The total market value of a company's outstanding shares. It helps categorize stocks into different segments like large-cap, mid-cap, and small-cap, which can have different growth potentials.

3. **Price-to-Earnings (P/E) Ratio**: Indicates the market's expectations of a company's growth potential. A higher P/E ratio might suggest that the stock is expected to have higher future growth.

4. **Debt-to-Equity Ratio**: This financial leverage ratio compares a company's total liabilities to its shareholder equity. It can indicate how a company is financing its growth and how risky its capital structure is.

5. **Return on Equity (ROE)**: A measure of financial performance calculated by dividing net income by shareholder equity. It shows how effectively management is using a company’s assets to create profits.

6. **Beta**: Measures a stock's volatility relative to the overall market. Growth stocks often have higher beta values, indicating more volatility.

7. **Revenue Growth**: The increase in a company's sales over a period. Consistent revenue growth is a sign of a potentially successful and expanding business.

8. **Dividend Yield**: The ratio of a company's annual dividend compared to its share price. Growth stocks often reinvest profits rather than paying high dividends.

9. **Price-to-Book (P/B) Ratio**: Compares a company's market value to its book value. A lower P/B ratio could indicate an undervalued stock with growth potential.

When engineering these features for Tenny, it's important to consider how they interact with each other and their relevance over time. Additionally, feature normalization or standardization might be necessary to ensure that Tenny can effectively learn from these features without bias toward those with larger scales.

Many view technical indicators such as RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and Bollinger Bands as key elements for classifying stocks. However, it's important to note that these indicators generally align more with short-term trading tactics, which aren't the main concern of this project. For this reason, I've deliberately left them out of even the basic list of features. I'm quite familiar with these indicators – in fact, I know them well enough to consciously choose not to employ them in my long-term investment strategies. These indicators are often like enticing statistical toys, providing a false sense of security and control.

### My Criteria for Growth Stocks

Again, these are a subset of features, based on my investment philosophy. 

Each offers unique insights into a company's financial health and growth potential. Here's why each of these indicators is significant in my opinion:

1. **Earnings Growth**: 
   - **Significance**: Earnings growth is a direct indicator of a company's profitability and its capacity to increase earnings over time. High earnings growth often signals a company's success in expanding its business, improving efficiency, or entering new markets.
   - **Why Use It**: It reflects the company’s potential for future expansion and is a key metric investors look for in growth stocks.
   - **My Take**: Growth is the name of the game. If a company is not growing, it's not a growth stock. It's that simple. Growing stocks should have growing earnings. If not, they're not growing. The sole exception to this straightforward rule is when you spot an early-stage company that has a promising product or service but hasn't yet reached profitability. Here, you're essentially gambling on the company's future potential rather than its present earnings. It's a high-risk move, but one that can yield substantial rewards if the company thrives – similar to investing in Tesla during its nascent phase. If you decide to go all-in with such companies, the outcome is binary: you either end up filthy rich or just plain filthy. It's all about your risk tolerance. If you're comfortable with high-risk, high-reward investments, then go for it. If not, then stick to the safer options.


2. **Price-to-Earnings (P/E) Ratio**:
   - **Significance**: The P/E ratio compares a company's stock price to its earnings per share, providing insight into how the market values the company’s earnings growth potential.
   - **Why Use It**: A higher P/E ratio may indicate that investors expect higher earnings growth in the future, making it a crucial metric for growth stocks.
   - **My Take**: The P/E (Price-to-Earnings) ratio is indeed a general indicator, not an absolute rule. It serves as a useful starting point, but it's important to delve deeper to grasp the underlying factors influencing the ratio. For example, a high P/E ratio might indicate a company's strong growth prospects, but it could also result from a recent spike in stock price. In the latter scenario, the high P/E ratio may not be sustainable, potentially signaling an overvalued stock. Additionally, companies in turnaround phases or early-stage growth stocks often exhibit extremely high or even negative P/E ratios. Therefore, it's crucial not to rely solely on P/E ratios; they are just one piece in the puzzle of many indicators. 

3. **Return on Equity (ROE)**:
   - **Significance**: ROE measures how effectively a company uses its equity to generate profits. It’s a gauge of a company’s profitability and efficiency.
   - **Why Use It**: High ROE values are often found in successful growth companies that efficiently generate profits from their assets.
   - **My Take**: The same rule for P/E ratios applies here, but in reverse. A low ROE might indicate a company's inefficiency or poor profitability, but it could also be a result of a recent dip in earnings. Therefore, it's important to consider the context and underlying factors influencing the ROE. Extremely high ROE values might also be a red flag, possibly indicating a company's excessive use of debt to finance its growth. Over the long haul, extreme values are often unsustainable, so it's important to consider the context and underlying factors influencing the ROE. I've never observed a scenario where a company maintains an ROE of around 50% for an extended period. Such high levels of ROE are generally not sustainable over the long term. You're noticing the pattern here, aren't you? From an object-oriented perspective, it should be becoming clear.

4. **Beta**:
   - **Significance**: Beta measures a stock's volatility relative to the overall market. A higher beta indicates that a stock’s price is more volatile.
   - **Why Use It**: Growth stocks often have higher beta values, reflecting their higher risk and potential for higher returns.
   - **My Take**: Exactly, the underlying principle remains the same, with just subtle differences in nuances – all fitting within the framework of object orientation. Think about it this way: as a stock's valuation approaches the status of being the world's most valuable company, its beta typically starts to decrease significantly. Why is that? It's because more market participants start buying its shares, which in turn normalizes the stock's price volatility. Investing in such a stock becomes akin to investing in the index against which it's benchmarked. That's the essence of what Beta represents. It's a metric that measures a stock's volatility relative to the overall market. I typically don't label stocks with a beta that matches or is lower than the index as growth stocks; they tend to exhibit features more akin to stalwarts. So, why do I sometimes hold onto these stocks? They act as cash equivalents. Stock investors generally aren't fond of holding cash. In situations where you're uncertain about the current composition of your portfolio, stalwarts serve as a viable alternative to cash holdings. Then, when you identify a definitive trend in growth stocks, you can simply sell the stalwarts and invest in growth stocks. This fluidity is part of the stock market's charm; it's always changing, always dynamic.

5. **Revenue Growth**:
   - **Significance**: This metric shows the rate at which a company's revenue is increasing. Steady revenue growth is a sign of expanding business operations and market share.
   - **Why Use It**: Consistent revenue growth is a hallmark of growth companies, as it often precedes earnings growth and can drive stock prices up.
   - **My Take**: If I need to reiterate my perspective on this, it means the message isn't quite sinking in. It's just another object, for heaven's sake! Everything, absolutely everything, is an object. Remember that key principle of the object-oriented perspective.

6. **Dividend Yield**:
   - **Significance**: Dividend yield is the ratio of a company’s annual dividend compared to its share price. While growth stocks are not typically known for high dividends, this metric can still be relevant.
   - **Why Use It**: A lower dividend yield might indicate that the company is reinvesting profits back into growth initiatives rather than distributing them to shareholders.
   - **My Take**: This is a tricky one. Think of it as an object inheriting from the abstract class of indicators, but with an attitude of its own, showcasing classic polymorphism. Generally, growth companies don't distribute dividends. If they start to, it often raises eyebrows. Why opt for dividend payments when the return on equity is surging? It doesn't quite add up. However, remember the fundamental concept: everything is an object. Stocks evolve, and unlike living organisms like humans, they can sometimes revert to earlier stages of their lifecycle. Or, in some cases, human nature might drive a desire for a share of the profits, regardless of the company's growth potential. Insiders, like CEOs and executives, selling their shares? That's not always a cause for alarm; they might have personal reasons that don't necessarily signal trouble. However, a company that suddenly decides to issue dividends, particularly in the face of declining earnings growth, is a major warning sign. It could indicate a company on the brink of stunted growth. This is more than just a red flag; it's a glaring signal that the company's growth trajectory may be faltering.
   
By focusing on these indicators, I am targeting key attributes that collectively paint a picture of a company's growth potential. They provide a comprehensive view, covering profitability, market valuation, financial efficiency, volatility, revenue trends, and dividend policy, all crucial in identifying promising growth stocks.

Keep in mind, as I mentioned earlier, these criteria are just a subset tailored for this project. In my personal investment strategies, I employ a far more comprehensive set of criteria.

But again, these are my perspectives, not necessarily yours. You might adopt some of these views, but you'll also cultivate your own unique insights. That’s the beauty of object orientation; it offers a flexible framework that can be tailored to your individual outlook and approach: inheritance and polymorphism in action.

### Additional Features for Our Dataset

![excel-data.png](images%2Fexcel-data.png)

Given the streamlined focus of our example code on the 12 most renowned tech companies (including ones for predictions) and aiming for simplicity,let's consider what we already have and what we should add to our dataset.

1. **Historical Earnings Data**:
   - Since we already have **Net Income / CAGR 5Y** and **Normalized Net Income / CAGR 5Y**, these can serve as a proxy for **Earnings Growth**. They provide a sense of how the company's earnings have changed over a five-year period.

2. **Stock Price History**:
   - With **Normalized Price** in our dataset, we can assess price trends over time. For calculating **Beta**, we need data on a relevant market index to compare the stock’s volatility against, but for the sake of simplicity, we can simply pull Beta from the source.

3. **Price/Earnings - P/E (LTM)**:
   - This feature is already in our dataset and is essential for analyzing the **P/E Ratio**. It helps gauge market expectations of a company's growth potential.

4. **Return On Equity %**:
   - This is also already included in the dataset. **ROE** is a key metric for evaluating a company’s profitability and efficiency in using its equity.

5. **Dividend Information**:
   - If our focus is on tech companies, many of which do not pay high dividends, we might choose to omit detailed dividend data for simplicity. However, a basic understanding of **Dividend Yield** can still be valuable, so we consider including it if readily available.

6. **Sector and Industry Classification**:
   - Since our dataset is focused on tech companies, this classification may not add significant value to our analysis, given the sector homogeneity.

7. **Total Revenues / CAGR 5Y**:
   - This existing feature helps in understanding **Revenue Growth**, which is a critical component of assessing a company's expansion and market share increase.

Considering the scope of our project, these features should provide a robust foundation for classifying stocks into 'Growth', 'Stalwart', and 'Speculative' categories. They capture key aspects of each company's financial performance and market behavior, offering a focused yet comprehensive view suitable for our example code.

However, adding both Beta and Dividend Yield (D/Y) to the dataset will enhance its capability for classifying stocks into the 'Growth', 'Stalwart', and 'Speculative' categories. 

_Market Capitalization_ isn't included in our current dataset. Market capitalization is crucial for assessing a company's size and stability, but since all these companies are in the large-cap category, its impact might not be as pivotal for our analysis. Nonetheless, it's a factor worth considering for a more exhaustive dataset. 

With these three additional metrics, Beta, Dividend Yield, Market Caps, our dataset becomes more comprehensive in capturing the essential aspects of stock performance and company policies, crucial for your classification objectives. Beta addresses the risk and return aspect, while Dividend Yield provides insight into the company's financial strategy regarding earnings distribution. Together, they contribute significantly to a nuanced analysis of stocks based on your investment criteria.

Note that for the purpose of our classification model, having current Beta values should be sufficient, especially considering the context of the project as an example code using real-world data from renowned tech companies. 

1. **Relevance to Current Market Conditions**: Current Beta values reflect the most recent volatility of the stocks relative to the market, which is more relevant for making current investment decisions. Stock volatility can change over time due to various factors, so the most recent Beta value is typically the most pertinent for your analysis.

2. **Simplicity and Focus**: Since our project is intended as an example rather than a comprehensive production-grade model, focusing on current Beta values keeps the analysis straightforward and manageable. This approach aligns well with the goal of creating a clear and concise educational example.

3. **Practicality for Classification**: Classification models are often designed to make assessments based on the current state of the data. In this case, the current Beta value provides a snapshot of the stock's present risk profile, which is crucial for classifying stocks as 'Growth', 'Stalwart', or 'Speculative'.

4. **Data Availability and Efficiency**: If only current Beta values are readily available, using them is a practical decision. It avoids the complexity and additional data processing that would be required to incorporate historical Beta values.

Adopting the method of filling only the latest quarter's data with the current Beta value in our quarterly dataset strikes a good balance between practicality and precision in depicting stock volatility. This approach respects Beta's time-sensitive nature while simplifying its integration into the dataset, which is mainly concerned with quarterly financial performance. Practically, this means for each stock in the dataset, the Beta column will be blank for all quarters except the most recent one. This quarter will display the current Beta value, mirroring the stock's latest risk stance in relation to market volatility. This method effectively encapsulates the stock's recent risk evaluation, avoiding the complication of incorporating possibly irrelevant or mismatched historical Beta data.

By using this strategy, we make sure that the model and any analyses that follow acknowledge the importance of the Beta value as a current indicator, pertinent to the existing market conditions. It streamlines data handling by focusing on the latest risk evaluation instead of tracing each stock's historical volatility. This approach also dovetails nicely with the educational and demonstrative goals of our project, offering a straightforward and clear depiction of how market risk is assessed in conjunction with other financial parameters.

Here's a basic script that retrieves Betas from Yahoo Finance for the 12 companies in our dataset, which also includes the stocks we'll be predicting:

```python
import yfinance as yf

# List of tickers for which you want to pull Beta values
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']

# Initialize a dictionary to store Beta values
beta_values = {}

# Retrieve and store Beta for each ticker
for ticker in tickers:
    stock = yf.Ticker(ticker)
    beta_values[ticker] = stock.info.get('beta')

# Print the Beta values
for ticker, beta in beta_values.items():
    print(f"{ticker}: Beta = {beta}")

# As of 2024-01-15
# AAPL: Beta = 1.29
# MSFT: Beta = 0.876
# AMZN: Beta = 1.163
# TSLA: Beta = 2.316
# GOOGL: Beta = 1.054
# META: Beta = 1.221
# NVDA: Beta = 1.642
# INTC: Beta = 0.995
# AMD: Beta = 1.695
# ADBE: Beta = 1.33
# NFLX: Beta = 1.283
# AVGO: Beta = 1.241
```

Can you guess why Microsoft's Beta is significantly lower compared to other companies, even lower than Apple's? Speculate on the reasons, especially considering Microsoft's status as one of the most valuable companies globally. 

Refer back to my take on Beta: "Exactly, the underlying principle remains the same, with just subtle differences in nuances – all fitting within the framework of object orientation. Think about it this way: as a stock's valuation approaches the status of being the world's most valuable company, its beta typically starts to decrease significantly. Why is that? It's because more market participants start buying its shares, which in turn normalizes the stock's price volatility. Investing in such a stock becomes akin to investing in the index against which it's benchmarked. That's the essence of what Beta represents. It's a metric that measures a stock's volatility relative to the overall market. I typically don't label stocks with a beta that matches or is lower than the index as growth stocks; they tend to exhibit features more akin to stalwarts. So, why do I sometimes hold onto these stocks? They act as cash equivalents. Stock investors generally aren't fond of holding cash. In situations where you're uncertain about the current composition of your portfolio, stalwarts serve as a viable alternative to cash holdings. Then, when you identify a definitive trend in growth stocks, you can simply sell the stalwarts and invest in growth stocks. This fluidity is part of the stock market's charm; it's always changing, always dynamic." 

We revise the script to add Beta to the dataframes. All the other indicators are pulled from the existing source: Kyofin.

Here's the complete code for retrieving the Beta info and adding it to the dataframes:

```python
import yfinance as yf
import pandas as pd
import numpy as np
import os

# List of tickers for which you want to pull Beta values
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']
data_folder = "./data"
enhanced_data_folder = "./enhanced-data"

# Initialize a dictionary to store Beta values
beta_values = {}

# Retrieve and store Beta for each ticker
for ticker in tickers:
    stock = yf.Ticker(ticker)
    beta_values[ticker] = stock.info.get('beta')

# Update each company's dataframe with the Beta value and save as a new CSV
for ticker in tickers:
    file_name = f"enhanced-raw-data-{ticker.lower()}.csv"
    file_path = os.path.join(data_folder, file_name)

    # Read the company's dataframe
    df = pd.read_csv(file_path)

    # Create a new row for Beta
    beta_row = pd.DataFrame([['Beta'] + [np.nan] * (len(df.columns) - 2) + [beta_values[ticker]]], columns=df.columns)

    # Append the new row to the DataFrame using pd.concat()
    df = pd.concat([df, beta_row], ignore_index=True)

    print(df.tail())

    # Save the updated dataframe as a new CSV file
    new_file_name = f"beta-{file_name}"
    new_file_path = os.path.join(enhanced_data_folder, new_file_name)
    df.to_csv(new_file_path, index=False)


# Note: As of 2024-01-15, Beta values are as printed above
# This code assumes that the CSV files exist in the specified folder and follows the naming pattern mentioned.

# As of 2024-01-15
# AAPL: Beta = 1.29
# MSFT: Beta = 0.876
# AMZN: Beta = 1.163
# TSLA: Beta = 2.316
# GOOGL: Beta = 1.054
# META: Beta = 1.221
# NVDA: Beta = 1.642
# INTC: Beta = 0.995
# AMD: Beta = 1.695
# ADBE: Beta = 1.33
# NFLX: Beta = 1.283
# AVGO: Beta = 1.241
```

![beta-pulling.png](images%2Fbeta-pulling.png)

Beta rows are added to the dataframes, and the updated dataframes are saved as new CSV files.

## Adding Target Labels to the Dataset

In order to incorporate labels like 'growth', 'stalwart', and 'other' into our dataframes for Tenny's learning process, we will have to manually categorize each stock using our set criteria and comprehension of these categories. Given that we're adopting a tailored approach and these classifications carry a degree of subjectivity, it's crucial that your assessments are grounded in a clear and consistent methodology.

Let's assume you are the one manually classifying these stocks. Here's a step-by-step process to add these labels:

1. **Define Your Classification Criteria**: Clearly define what characteristics or metrics qualify a stock as 'growth', 'stalwart', or 'other'. This could be based on the indicators you've chosen, such as earnings growth, P/E ratio, ROE, Beta, Revenue Growth, and Dividend Yield.

2. **Review Each Stock**: For each stock in your dataset, review its historical and current performance based on your chosen metrics. You might need to look at trends, compare metrics against industry averages, and consider the company's overall market position.

3. **Classify and Label Each Stock**: Based on your review, classify each stock as 'growth', 'stalwart', or 'other'. 

4. **Add Labels to DataFrame**: Create a new row in each stock's DataFrame labeled 'Classification' or something similar. In the column for the most recent quarter (or the relevant period for your analysis), add the label you've assigned ('growth', 'stalwart', or 'other'). 

5. **Save the Updated DataFrame**: After adding the classification labels, save the updated DataFrame as a new CSV file or overwrite the existing one if that fits your workflow.

Here's a snippet of how you might modify the DataFrame:

```python
# Example for a single stock DataFrame
classification = 'growth'  # Replace with the actual classification for this stock

# Add a new row for classification
classification_row = pd.DataFrame([['Classification'] + [np.nan] * (len(df.columns) - 2) + [classification]], columns=df.columns)

# Append the new row to the DataFrame
df = pd.concat([df, classification_row], ignore_index=True)

# Save the updated DataFrame
df.to_csv("updated_stock_data.csv", index=False)
```

Repeat this process for each stock in your dataset. This approach will allow Tenny to learn the classifications based on the historical and current data of each stock, combined with your expert judgment.

This is exactly what I plan to do with **_MY_** dataset in the upcoming chapter. I'm intentionally stressing the word 'my' to underscore that these criteria are mine, shaped by my own investment philosophy.

I'll leave you with some food for thought based on our discussions so far. I encourage you to step outside the usual boundaries you associate with these concepts. Consider how you can adapt and apply these ideas across various fields. This is what I mean by expanding your horizons.

Every damn thing is an object.