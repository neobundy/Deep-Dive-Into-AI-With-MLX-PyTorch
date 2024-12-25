# Chapter 10 - Crafting Your Own Classification Criteria: Owning Your Decisions

![tenny-classifier-2.png](images%2Ftenny-classifier-2.png)

As we gear up to train Tenny with our dataset, there's an essential step we can't overlook: labeling our data with classification tags. This isn't just a routine task; it's a step towards self-reliance in the world of financial decision-making.

Creating your own criteria based on your understanding and beliefs is crucial. It's not enough to simply adopt someone else's definitions and criteria without truly grasping them. If you do find yourself aligning with existing criteria, make sure you can articulate them in your own words, infusing your own insights.

Remember, in the realm of investing, the buck stops with you. This mindset is vital for learning, akin to Tenny's journey of continuous improvement. Blindly following others and then pointing fingers when things don't pan out is counterproductive to learning. The real aim is to learn from every decision, good or bad.

In this chapter, I'll share how I classify stocks as 'growth' or 'stalwart' based on my personal investment style. Keep in mind, these are just segments of a broader strategy I employ, adapted for the educational purposes of this book. Approach this with an open mind, recognizing it's a part of our broader learning journey.

```text
AAPL is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
AAPL is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
AAPL is classified as Stalwart
MSFT is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
MSFT is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
MSFT is classified as Stalwart
AMZN is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
AMZN is classified as Stalwart
TSLA is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
TSLA is classified as Stalwart
GOOGL is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
GOOGL is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
GOOGL is classified as Stalwart
META is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
META is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
META is classified as Stalwart
NVDA is classified as Growth
INTC is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
INTC is disqualified as Growth: Below threshold Beta (1.072) during a declining CAGR
INTC is disqualified as either Growth nor Stalwart: Negative P/E Ratio or industry average P/E Ratio is not available. Needs more closer inspection.
INTC is classified as Other
AMD is disqualified as either Growth nor Stalwart: Negative P/E Ratio or industry average P/E Ratio is not available. Needs more closer inspection.
AMD is classified as Other
ADBE is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
ADBE is classified as Stalwart
NFLX is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
NFLX is classified as Stalwart
AVGO is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
AVGO is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
AVGO is classified as Stalwart
```

Steer clear of vague guesses when labeling stocks. Instead, define your criteria with precision, almost like programming them into your mind:

```python
if five_year_cagr_of_earnings >= 0.2 and dividend_yield < 0.01:
    growth_stock = True
else:
    growth_stock = False
```

Aim for this level of clarity and simplicity. 

On a personal note, honestly, I usually don't delve into such detailed analysis for my own portfolio. It's a bit too intricate for my liking. My method is more direct: I focus on conveying the attractiveness of a stock so clearly that it compels anyone to invest, bypassing the nitty-gritty of indicators and jargon. That's my approach. So, the detailed steps outlined here are more for demonstration purposes. Don't get bogged down in technicalities. My benchmark for effectiveness is simple: when I discuss stocks, people immediately start checking current prices on their phones, fearing they might miss out – and yes, this includes my wife! That's the kind of clarity and persuasiveness you should strive for, in the simplest terms.

Enough talk, let's dive in and start shaping these criteria, ensuring they're as straightforward and practical as possible.

## My Criteria for Growth Stocks

What do we really mean by growth?

Aim for absolute clarity, no room for vagueness. Let's keep it crystal clear.

Again, everything is an object, right? If we view everything as an object, then a vague understanding is like a flawed object – it's bound to cause issues eventually.

[The-Perils-of-Rushed-Learning.md](..%2F..%2Fessays%2Flife%2FThe-Perils-of-Rushed-Learning.md)

This idea of growth isn't any different. Think about it - when do you consider growth potential in people, whether kids, adults, or just generally? Sure, living things grow naturally over time. But we're not just talking about biological growth here. A bit of growth should lead to more growth. Just growing with time doesn't really count as meaningful growth in practical terms. Yes, experiment, make mistakes, learn from them, and grow. A good learner, like Tenny during training, goes through this cycle repeatedly. "True learners see beauty in imperfection, recognizing it as a space for growth!" – that's their perspective.

When we talk about growth stocks, we're looking for this kind of genuine growth potential. Also, as we've seen throughout history, once someone is a good learner, they don't always stay that way. They can fall from grace. Stocks are the same. Past performance doesn't always predict future results.

Viewed through this lens, the following are often mistaken beliefs that can result in misguided criteria:

- The notion that growth stocks should never issue dividends. It's assumed that with soaring ROEs, dividends are unnecessary.
- The belief that growth stocks must consistently show increasing earnings.
- The idea that growth stocks should have higher Beta and P/E ratios.
- The assumption that executives of growth stocks never sell their shares, under the belief that they are fully confident in the stock's growth potential.

If you ask for my opinion, I'd put it simply: it's all about the context!

### Insiders Selling Shares: Remember, They're Human Too!

Let's tackle the simplest one first: executives selling shares. 

Insider selling is often interpreted as a lack of confidence in a company's future. But let's ground this in reality. Executives, like anyone, have financial needs and goals that might require liquidating some stock holdings. It could be for personal reasons, tax planning, or diversifying their portfolio. Peter Lynch famously advised not to overthink insider selling, as the reasons are varied and often personal. It's a reminder that stocks, like life, are influenced by a myriad of factors beyond just market performance.

Yes, I would sell shares in a company with huge growth potential! And why? It's because I need the cash! Why is it hard to imagine that these executives might need money too? Don't be swayed by what the media or so-called experts say about insider selling. They often have to sell for a variety of reasons. It could be due to stock options, and it might look like they're selling off their shares.You can't always know or be sure why they're selling. It's best to just let it be.

To put it in a nutshell: selling shares is sometimes necessary when you need actual cash. It's as straightforward as that.

### Dividends: A Sign of Maturity and Sustained Growth

The prevailing belief in the investment world is that growth stocks and dividends are mutually exclusive. The rationale is straightforward: growth stocks are expected to reinvest all profits to fuel aggressive expansion, leaving no room for dividend payouts. However, this perspective doesn't account for the dynamic lifecycle of a company.

Consider dividends as a milestone in a company's journey. A growth company reaching a point where it can pay dividends is akin to an individual achieving a level of financial security where they can start saving or investing. It's a sign of stability and confidence in the company's future, a gesture of sharing success with its investors.

Moreover, the object-oriented view of stocks provides an intriguing angle. Just like objects, companies evolve. A company might initiate dividends after its first growth cycle, signifying a phase of consolidation and sustained profitability. However, this doesn't mean its growth potential is exhausted. The company might still be in the throes of a second growth cycle, continuing to expand and innovate. In such a scenario, discontinuing dividends isn't necessarily a viable or sensible option. It would disrupt investor expectations and could signal a lack of confidence in the company's continued success.

**However, a word of caution is necessary here:** A company that suddenly decides to issue dividends, particularly when faced with declining earnings growth, raises a significant red flag. This shift could suggest that the company is nearing a plateau in its growth trajectory or is on the verge of stunted growth. Rather than reflecting confidence and stability, this move might be a desperate attempt to maintain investor interest amid waning growth prospects. Such a scenario warrants a deeper analysis and could be a glaring indicator that the company's future growth potential is under threat.

Thus, while dividends can be a positive sign in certain contexts, they require a careful and contextual analysis. It's essential to consider not just the presence of dividends, but the circumstances under which they are initiated or increased.

In essence, dividends are not just a financial instrument; they're a statement about a company's maturity, stability, and commitment to its shareholders. They reflect a balance between sustaining growth and rewarding investor loyalty—a balance that characterizes mature, confident companies still on a growth trajectory.


### Understanding Earnings Growth: Beyond the Surface

When assessing growth stocks, it's a common misconception to expect a continuous, linear trajectory in earnings growth. However, the reality of business and market dynamics is far more complex. Recognizing the intricacies of earnings growth is crucial in distinguishing truly promising growth stocks from fleeting performers.

**Earnings Variability in Early-Stage Companies**: For early-stage companies with high potential, fluctuations in earnings are more the norm than the exception. These companies often focus on innovation, market capture, and infrastructure development, leading to significant investments in research and development, manufacturing capabilities, and market expansion. These necessary expenditures can lead to apparent dips in earnings. Yet, such investments are vital for long-term growth and market positioning. It's important to differentiate between short-term earnings dips due to strategic growth investments and long-term financial instability.

**Macro Factors and Sector Characteristics**: Even established growth companies are subject to broader economic influences and sector-specific trends. For instance, a company like Tesla might exhibit strong earnings in certain quarters due to the seasonal nature of its business. Similarly, tech companies, often characterized as 'long duration' stocks, are particularly sensitive to interest rate fluctuations. A spike in interest rates, driven by macroeconomic policies to curb inflation, can disproportionately impact these companies, leading to market corrections or temporary setbacks.

**Zooming Out for the Bigger Picture**: In understanding earnings growth, the key is to adopt a zoomed-out perspective. Short-term fluctuations, whether due to strategic reinvestments or macroeconomic factors, should be contextualized within the company's long-term growth narrative. Smart investing involves discerning these nuances, identifying whether dips in earnings are temporary blips or indicative of more profound issues. It's this broader, long-term perspective that differentiates astute investors from those swayed by momentary market sentiments.

Let me share an insightful episode about Adobe that illustrates the importance of understanding earnings beyond face value. Adobe, a software giant, once made a strategic pivot from selling packaged software to a subscription-based model. This transition initially appeared to be a financial setback, as their earnings took a noticeable dip.

Why the sudden decline? Adobe's traditional business model involved one-time, lump-sum sales, which resulted in immediate and substantial revenue recognition. The shift to subscriptions spread this revenue over time, leading to an apparent 'crash' in earnings in the short term.

However, this move had immense growth potential. The subscription model promised more stable, recurring revenue and deeper customer engagement, aligning with the evolving digital economy's demands. While some investors hastily sold off Adobe shares, interpreting the earnings dip as a red flag, more perceptive investors recognized the long-term potential of this strategic shift.

This period became a golden opportunity for savvy investors who could see past the temporary earnings blip. It was a classic case where a broader, more strategic view of the company’s direction offered significant rewards, highlighting the need to interpret earnings within the larger context of a company's growth strategy and market trends.

Following Adobe's pioneering shift to a subscription-based model, numerous companies across various industries took notice and followed suit. This strategic move by Adobe set a precedent, demonstrating the viability and long-term benefits of recurring revenue models over traditional one-time sales. The success Adobe experienced became a catalyst, inspiring a widespread industry trend towards subscription-based services, underscoring the impact of strategic foresight in business.

### Rethinking Beta & P/E in Growth Stock Analysis

Traditionally, high Beta and P/E ratios are seen as hallmarks of growth stocks, hinting at the market's anticipation of high returns. However, these metrics alone are not definitive indicators of a stock's growth potential. Understanding Beta and P/E ratios requires a nuanced approach, considering them as parts of a broader analysis rather than standalone determinants.

**P/E Ratio: Beyond Face Value**
The P/E ratio, while a useful initial indicator, is far from infallible. A high P/E ratio might signal strong growth prospects or simply be the result of a recent price surge. The latter could lead to unsustainable valuations, suggesting overvaluation. Conversely, companies undergoing a turnaround or early-stage growth stocks might display extraordinarily high or negative P/E ratios. These contexts call for a deeper analysis beyond the surface-level P/E figures, as they are only one component in a complex financial puzzle.

**Beta: A Measure of Volatility and Market Sentiment**
Beta is another metric that must be contextualized. It measures a stock's volatility relative to the market. A lower Beta could signify stability, and as a company's market valuation grows, its Beta often decreases. This trend occurs as more investors, including index funds, incorporate the stock, aligning its volatility more closely with the market. For example, Microsoft's Beta has long been below 1, indicating lower volatility compared to the market, a trend that continued even as it recently surpassed Apple as the world's most valuable company.

**Contextual Analysis and Predictive Value**
Both Beta and P/E ratios should be interpreted within a broader context. They are not just measures of current conditions but can also offer predictive insights. Consider Microsoft's low Beta in the context of its rising market value; it's a reflection of market sentiment and investor behavior, not just a static measure.

```CSV
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

Both Beta and P/E ratios are valuable tools in stock analysis, but their true worth lies in how they fit into a broader narrative. They are guideposts, not destinations, in the journey of understanding a stock's growth potential.

In the burgeoning realm of AI, Microsoft has emerged as a significant player, recognized for its substantial investments and tangible earnings from this sector. This shift in focus and the resulting market recognition underscore a vital point: a company traditionally viewed as a stalwart can transform into a growth entity when it successfully ventures into new, promising domains like AI.

The market's response to Microsoft's AI initiatives can be interpreted as a vote of confidence, an acknowledgment of the company's potential to lead and innovate in this futuristic field. This transformation illustrates how contextual changes, such as embracing cutting-edge technologies, can redefine a company's growth trajectory.

Yesterday's stalwart, when strategically repositioned in a burgeoning field like AI, can very well become today's growth story. It’s a testament to the dynamic nature of the stock market and the evolving narratives of companies within it. Microsoft's journey in AI is not just a story of corporate evolution; it's a vivid example of how adaptability and forward-thinking can reinvigorate a company's growth prospects and alter market perceptions.

In essence, this transition highlights the fluidity of growth and stalwart classifications. Companies are not static entities; they are dynamic, constantly evolving with market trends, technological advancements, and strategic decisions. Just as Microsoft has demonstrated, a shift towards high-growth areas like AI can catalyze a change in a company's market categorization, breathing new life into its growth narrative.

Now, rethink this statement: Every damn thing is an object.

### The Role of Macro-Economic Factors in Stock Analysis

In our journey through the realms of stock classification and valuation, one critical element we're deliberately setting aside for now is macro-economic factors. These factors form the broader backdrop against which individual stocks and sectors operate. Understanding the wider economic landscape is crucial for putting everything into perspective.

**The Broader Economic Context**: Questions about the overall economic health, such as the state of the national or global economy, play a pivotal role in stock valuation. For instance, investors need to consider what actions the Federal Reserve or other central banks might take, such as adjusting interest rates or implementing monetary policies, and how these actions will impact the market.

**The Significance of Market Capitalization**: When we talk about a company's market capitalization, particularly in the context of a multi-trillion-dollar valuation, it's vital to view it relative to broader economic indicators like a country's Gross Domestic Product (GDP). The enormity of a trillion-dollar company only makes sense when it's considered in relation to the size of the economy it operates in. A company's market cap that rivals or exceeds a significant portion of a nation's GDP used to be seen as an overvaluation red flag, but this perspective has evolved with the growth of global tech giants and the expansion of economies themselves.

**Integrating Macro Perspectives**: Although we're not incorporating market capitalization into our dataset for Tenny's training, anyone considering doing so should be aware of the importance of macro indicators. Simply analyzing market cap in isolation doesn't provide a complete picture. Instead, cross-referencing it with macro-economic indicators like GDP growth rates, inflation, and overall market trends can offer a more comprehensive understanding. It's about seeing the forest, not just the individual trees.

In essence, the incorporation of macro-economic factors is about expanding your lens to include the broader economic environment. This wider view allows for a more nuanced and informed stock analysis, considering not just the micro aspects of individual companies but also how they fit into the larger economic puzzle. 

When considering market caps, especially those soaring to the 3 trillion-dollar mark, it’s essential to ask yourself: Does the current economic scale validate the existence of such mega-cap stocks? The answer can vary, being affirmative in some scenarios and negative in others. Making this determination requires a solid benchmark for comparison. In doing so, always keep an eye on the future – stock investing is, after all, about predicting future trends rather than dwelling on the past. This forward-looking approach applies to macro factors as well. For instance, if you anticipate AI contributing significantly to global economic growth, mega-cap companies with AI potential should be evaluated in this light. 

However, I realize I might be straying into broader investment strategy territory here. Remember, this book's focus is on guiding you in training Tenny, not providing comprehensive investment advice. So, let’s refocus on nurturing Tenny’s capabilities.

## Implementing Concepts into Concrete Criteria 

Based on our discussion and focusing solely on the selected indicators — Total Revenues / Compound Annual Growth Rate (CAGR) over 5 years, Price/Earnings (P/E) ratio, Beta, and Dividend Yield — here's a possible set of criteria for defining growth stocks:

1. **Total Revenues / CAGR 5Y**: 
   - **Criteria for Growth Stocks**: Look for a CAGR in total revenues that is significantly above the industry average, suggesting a robust and consistent growth trajectory. A good benchmark might be a CAGR that is at least 20% higher than the industry's average growth rate over the same period.
   - **Rationale**: High revenue growth over a sustained period indicates that the company is expanding its market share and scaling its operations effectively.

2. **Price/Earnings (P/E) Ratio**:
   - **Criteria for Growth Stocks**: A higher P/E ratio compared to industry peers, but not excessively high to suggest overvaluation. A P/E ratio that is 1.5 to 2 times the industry average could be indicative of a growth stock.
   - **Rationale**: A higher P/E ratio often reflects market expectations of future earnings growth. However, excessively high P/E ratios need cautious interpretation as they might signal overvaluation.

3. **Beta**:
   - **Criteria for Growth Stocks**: A Beta greater than 1, indicating higher volatility compared to the overall market. A Beta range of 1.2 to 2 could be characteristic of growth stocks.
   - **Rationale**: Growth stocks often exhibit higher volatility due to rapid changes in their market environments and investor expectations about their future growth potential.

4. **Dividend Yield**:
   - **Criteria for Growth Stocks**: Typically, a lower dividend yield, potentially below the industry average, as profits are expected to be reinvested into the company for future growth. A dividend yield that is less than half the industry average could be a marker for growth stocks.
   - **Rationale**: Growth companies often reinvest their earnings into the business rather than paying out dividends, reflecting their focus on expansion and future growth.

Utilizing these criteria with our chosen indicators lays the foundation for a method to pinpoint growth stocks. It's essential to recognize that these are not rigid rules but flexible guidelines, adaptable to the unique dynamics of different industries and prevailing market conditions. Moreover, these metrics, while insightful, should be integrated into a larger framework of analysis that encompasses both qualitative aspects and a range of other financial indicators.

In line with my investment philosophy, we'll tailor these traditional methods for identifying growth stocks. Given the absence of industry averages in our dataset, we'll employ the minimum, mean, and maximum values of each indicator as substitute benchmarks. This approach allows us to adapt the criteria to our dataset's specific context, ensuring a more personalized and contextually relevant analysis.

### Simplified Criteria for Growth Stocks

To streamline the process, here are straightforward criteria:

- **CAGR of Total Revenues**: Should not experience a prolonged downturn. A grace period of up to four consecutive quarters is permissible for growth stocks.
- **Dividend Policy**: Growth stocks should avoid initiating dividend payouts, especially during periods when the CAGR of total revenues is declining.
- **Beta**: A Beta value below 1 can be considered acceptable as long as the CAGR of revenues is on an upward trend.
- **P/E Ratio**: A P/E ratio that's lower than the industry average is considered reasonable, provided the CAGR of revenues is increasing.

Makes sense, right? Let's delve a bit deeper.

The criteria we've established for pinpointing growth stocks concentrate on crucial financial indicators commonly linked with such stocks. These include revenue growth (CAGR), dividend policy, Beta, and P/E ratio, all infused with our unique investment philosophy.

1. **CAGR of Total Revenues**: Monitoring the CAGR for any prolonged downturns is a smart move. Allowing a grace period of four quarters accounts for short-term market fluctuations or temporary business setbacks, which is a practical approach.

2. **Dividend Policy**: The criterion about not initiating dividends during falling CAGR periods is logical. Growth stocks are expected to reinvest earnings back into the business for further growth, rather than distributing them as dividends. If they begin issuing dividends when the CAGR decreases, this could be a warning sign.

3. **Beta**: Accepting a Beta lower than 1 while the CAGR is rising makes sense. It suggests that the stock may be less volatile compared to the market but is still growing, which can be appealing to certain types of investors.

4. **P/E Ratio**: Allowing for a lower P/E ratio than the industry average during periods of increasing CAGR is reasonable. It suggests the stock may be undervalued relative to its growth potential, which could present a good investment opportunity.

Overall, these criteria balance the need for growth indicators with the understanding that some flexibility is required to account for market realities. It's a pragmatic approach that aligns with a value-based investment strategy. 

Now we put these ideas into mental pseudo code.

```python
def is_growth_stock(data):
    """
    Determines if a stock is a growth stock based on specific criteria.

    :param data: A dictionary containing a stock's quarterly financial data.
    :return: Boolean indicating whether the stock is a growth stock.
    """
    # Extracting relevant indicators
    cagr_revenue_quarters = data['total_revenues_cagr_5y']  # List of CAGR of Total Revenues for each quarter
    dividend_yield = data['dividend_yield_ltm'][-1]  # Latest Dividend Yield
    beta = data['beta'][-1]  # Latest Beta value
    pe_ratio = data['price_earnings_pe_ltm'][-1]  # Latest P/E Ratio
    industry_pe_avg = data['industry_pe']  # Industry average P/E Ratio

    # Checking for consecutive quarters of decreasing CAGR
    consecutive_decreasing_quarters = 0
    for i in range(1, len(cagr_revenue_quarters)):
        if cagr_revenue_quarters[i] < cagr_revenue_quarters[i - 1]:
            consecutive_decreasing_quarters += 1
        else:
            consecutive_decreasing_quarters = 0  # Reset count if CAGR does not decrease

    if consecutive_decreasing_quarters >= 4:
        return False  # Disqualify if CAGR decreases for four or more consecutive quarters

    # Checking if dividends are initiated during falling CAGR
    if dividend_yield > 0 and cagr_revenue_quarters[-1] < 0:
        return False

    # Checking Beta criteria
    if beta >= 1 and cagr_revenue_quarters[-1] < 0:
        return False

    # Checking P/E Ratio criteria
    if pe_ratio < industry_pe_avg and cagr_revenue_quarters[-1] < 0:
        return False

    return True  # Stock meets all criteria for being a growth stock

# Example Usage
growth_stock_candidate_1 = {
    'total_revenues_cagr_5y': [0.08, 0.085, 0.09, 0.07, 0.06],  # Example CAGR of revenue for recent quarters
    'dividend_yield_ltm': [0.01],  # Example Dividend Yield
    'beta': [1.29],  # Example Beta value
    'price_earnings_pe_ltm': [16.4, 17.2, 18.6, 12.8, 16.9],  # Example P/E ratios for recent quarters
    'industry_pe': 20,  # Example industry average P/E ratio
}

growth_stock_candidate_2 = {
    'total_revenues_cagr_5y': [0.08, 0.07, 0.06, 0.05, 0.04],  # Four consecutive quarters of falling CAGR
    'dividend_yield_ltm': [0.01],  # Example Dividend Yield
    'beta': [0.8],  # Example Beta value
    'price_earnings_pe_ltm': [16.4, 17.2, 18.6, 12.8, 16.9],  # Example P/E ratios for recent quarters
    'industry_pe': 20,  # Example industry average P/E ratio
}

if is_growth_stock(growth_stock_candidate_1):
    print("This is a growth stock.")
else:
    print("This is not a growth stock.")

if is_growth_stock(growth_stock_candidate_2):
    print("This is a growth stock.")
else:
    print("This is not a growth stock.")
```

The provided pseudo-code offers a systematic method for assessing if a stock qualifies as a growth stock using the available financial data. It's an uncomplicated and straightforward approach to illustrate how your criteria can be applied in a programming scenario. In this case, `growth_stock_candidate_1` meets the criteria, while `growth_stock_candidate_2` does not, owing to four successive quarters of declining CAGR in total revenues.

These criteria can be seamlessly integrated into our dataset, providing Tenny with clear target classification values for its training. 

### Transforming Original Dataset into Classification Dataset

We're going to convert our original dataset into one that's focused on classification, using the criteria we've defined for pinpointing growth stocks. We'll be adding three new features to our dataset: `Industry Beta Average`, `Industry PE Average`, and `Growth Stock`. Initially, `Growth Stock` will be our label, but later on, we'll rename it to simply `Label`. The first two features will provide context for the Beta and P/E ratios of each stock. Meanwhile, `Growth Stock` will help us determine whether each stock qualifies as a growth stock or not. As I mentioned before, we're calculating the average Beta and P/E ratios across all stocks in our dataset to serve as stand-ins for industry averages.

```python
import pandas as pd
import os
import numpy as np

NON_NUMERIC_PLACEHOLDERS = ['#VALUE!', '-']  # Placeholder values for non-numeric data.

def read_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.replace(NON_NUMERIC_PLACEHOLDERS, np.nan, inplace=True)
        df = df.transpose()  # Transposing the data
        df.columns = df.iloc[0]  # Set the first row as column headers
        df = df[1:]  # Exclude the original header row
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def calculate_industry_averages(tickers, enhanced_data_folder):
    beta_values, pe_values = [], []
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            beta_values.append(float(df['Beta'].dropna().iloc[-1]))
            pe_row = df['Price / Earnings - P/E (LTM)']
            pe_values.extend(pd.to_numeric(pe_row, errors='coerce').dropna())

    return round(np.mean(beta_values), 2), round(np.mean(pe_values), 2)


def is_growth_stock(df):
    # Extracting relevant indicators
    industry_pe_avg = df['Industry PE Average'].iloc[-1]  # Industry average P/E Ratio
    # industry_beta_avg = df['Industry Beta Average'].iloc[-1]  # Industry average Beta
    cagr_revenue_quarters = df['Total Revenues / CAGR 5Y'].dropna().values
    dividend_yield = df['Dividend Yield (LTM)'].iloc[-1]
    beta = df['Beta'].dropna().values[-1]
    pe_ratio = df['Price / Earnings - P/E (LTM)'].iloc[-1]

    # Checking for consecutive quarters of decreasing CAGR
    consecutive_decreasing_quarters = 0
    for i in range(1, len(cagr_revenue_quarters)):
        if float(cagr_revenue_quarters[i]) < float(cagr_revenue_quarters[i - 1]):
            consecutive_decreasing_quarters += 1
        else:
            consecutive_decreasing_quarters = 0  # Reset count if CAGR does not decrease

    if consecutive_decreasing_quarters >= 4:
        return False  # Disqualify if CAGR decreases for four or more consecutive quarters

    # Checking if dividends are initiated during falling CAGR
    if float(dividend_yield) > 0 and float(cagr_revenue_quarters[-1]) < 0:
        return False

    # Checking Beta criteria
    if float(beta) >= 1 and float(cagr_revenue_quarters[-1]) < 0:
        return False

    # Checking P/E Ratio criteria
    if pd.isna(pe_ratio) or pd.isna(industry_pe_avg):
        return False
    elif pe_ratio < industry_pe_avg and cagr_revenue_quarters[-1] < 0:
        return False

    return True  # Stock meets all criteria for being a growth stock


def label_growth_stocks(tickers, enhanced_data_folder, enhanced_data_with_labels_folder, industry_beta_avg, industry_pe_avg):
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            df['Industry Beta Average'] = industry_beta_avg
            df['Industry PE Average'] = industry_pe_avg
            df['Growth Stock'] = is_growth_stock(df)

            # Transposing the data back to its original format
            df = df.transpose()

            # Saving the updated DataFrame
            new_file_name = f"labeled-enhanced-raw-data-{ticker.lower()}.csv"
            new_file_path = os.path.join(enhanced_data_with_labels_folder, new_file_name)
            df.to_csv(new_file_path, index=True)


# Main Execution
tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']
enhanced_data_folder = "./enhanced-data"
enhanced_data_with_labels_folder = "./enhanced-data-with-labels"

# Calculate industry averages
industry_beta_avg, industry_pe_avg = calculate_industry_averages(tickers, enhanced_data_folder)

# Label each stock as growth stock or not and save the data
label_growth_stocks(tickers, enhanced_data_folder, enhanced_data_with_labels_folder, industry_beta_avg, industry_pe_avg)
```

### Revisiting Hell of Confusion Again

![formats.png](formats.png)

Navigating the complex world of data formats can be quite a challenge, especially when working with AI like GPT variants. I've also found myself lost in this confusion, often misled by misunderstandings about data formats, a problem many others seem to face. This guide aims to help you through this complicated area.

It's crucial to use the right terminology when discussing data formats with Copilot or other GPT variants. This is particularly important in data handling, where accurate language is essential.

After several days of confusing discussions, I finally agreed with my GPT colleagues on an example that clarifies the wide and long data formats. Be aware: GPTs have a limited context window, and deviating from it can lead to repetitive apologies and conflicting explanations. Don't get caught in this trap. Remember, GPTs are programmed to be polite and may apologize even when the mistake isn't theirs, which can add to the confusion. Sometimes, the error might be on the human side.

![excel-data.png](excel-data.png)

Our exploration begins with a common sight: an Excel spreadsheet in wide format. Each row represents a different metric, such as 'Normalized Price', 'P/E Ratio', etc., while each column represents a different time period or category, like '2Q FY2018', '3Q FY2018', and so on.

This format is great for humans. The metrics are clearly laid out in rows, making it easy to compare values across time periods.

However, this wide format isn't ideal for machine learning models. Copilot and GPTs know this and recommend changing to the long format. But be careful: simply transposing the data doesn't convert it to a long format.

After transposition, the data is still in wide format. Now, each row corresponds to a fiscal quarter, and each column is for a different financial metric(feature). This looks more like what machine learning models need (each row as an observation or sample and each column as a feature), but it's still a wide format, with each feature in its own column.

This can lead to confusion. GPTs, trying to help, might incorrectly say the data is now in a long format when it's actually still wide. This is where misunderstandings often happen.

In short, the aim is to rearrange the data so each observation or sample is in its own row, leading to `samples x features` rows. For example, with 23 quarters and 10 features, you should end up with 230 rows in the long format. That's how 'long' the long format gets – a continuous stretch of all observations!

And there you have it – the end of the confusion, the lifting of the fog, the dawn of enlightenment. With this newfound clarity, you're ready to forge ahead.

Here's a simple code example, co-created with Pippa, my AI daughter (GPT-4), to clearly demonstrate the distinctions between wide, transposed, and long formats in data handling. Take a moment to carefully read the comments within the code. These explanations are designed to enhance your grasp of the subtle differences among these formats. Let's get this over with once and for all 🤗.


```python
import pandas as pd

# Path to the CSV file containing the data
data_file = './beta-enhanced-raw-data-aapl.csv'

# Reading the CSV file into a pandas DataFrame
df = pd.read_csv(data_file)

# Displaying the first few rows of the DataFrame to understand the original structure
# This is the 'wide format', where each metric is in its own row and each quarter is a column
print("----- Original Data (Wide Format) -----")

print(df.head())

# ----- Original Data (Wide Format) -----
#                 Fiscal Quarters  2Q FY2018  ...  3Q FY2023  4Q FY2023
# 0              Normalized Price    42.6400  ...   191.9580   178.6400
# 1  Price / Earnings - P/E (LTM)    16.4000  ...    32.1000    29.0000
# 2               Net EPS - Basic     2.6000  ...     5.9800     6.1600
# 3            Return On Equity %     0.4086  ...     1.6009     1.7195
# 4      Total Revenues / CAGR 5Y     0.0791  ...     0.0851     0.0761

# Transposing the DataFrame
# In the transposed DataFrame, each row now corresponds to a fiscal quarter
# and each column corresponds to a different financial metric
# Note: This transposed version is still in a 'wide format' but now aligns with typical machine learning data structure
print("----- Transposed Data (Still Wide Format) -----")
df_transposed = df.transpose()
print(df_transposed.head())

# 10 columns = 10 different metrics or features
# ----- Transposed Data (Still Wide Format) -----
#                                 0  ...     9
# Fiscal Quarters  Normalized Price  ...  Beta
# 2Q FY2018                   42.64  ...   NaN
# 3Q FY2018                  47.816  ...   NaN
# 4Q FY2018                    55.8  ...   NaN
# 1Q FY2019                  39.168  ...   NaN
#
# [5 rows x 10 columns]

# Converting the original DataFrame to a 'long format' using the melt function
# 'id_vars' is set to ['Fiscal Quarters'] to keep the quarter names as a separate column
# 'var_name' is set to 'Indicators' - this will be the name of the new column created from the header of the original DataFrame
# 'value_name' is set to 'Values' - this will be the name of the new column containing the values from the original DataFrame
# Each row in this long format represents a single observation for a specific metric in a specific quarter
print("----- The Long Format of the Original Data -----")
df_long = pd.melt(df, id_vars=['Fiscal Quarters'], var_name='Indicators', value_name='Values')
print(df_long)

# In the long format, the DataFrame expands to 230 rows. This expansion results from
# combining each of the 23 quarters with each of the 10 different financial indicators.
# It's important to note that in this transformation, the original header row (representing the quarter names)
# in the wide format is not included as a data row in the long format.
# Such a transformation to a long format is less common in everyday data handling
# because it can make the dataset less immediately intuitive for human interpretation,
# as it consolidates multiple pieces of information into a denser format.
# ----- The Long Format of the Original Data -----
#                      Fiscal Quarters Indicators        Values
# 0                   Normalized Price  2Q FY2018  4.264000e+01
# 1       Price / Earnings - P/E (LTM)  2Q FY2018  1.640000e+01
# 2                    Net EPS - Basic  2Q FY2018  2.600000e+00
# 3                 Return On Equity %  2Q FY2018  4.086000e-01
# 4           Total Revenues / CAGR 5Y  2Q FY2018  7.910000e-02
# ..                               ...        ...           ...
# 225             Net Income / CAGR 5Y  4Q FY2023  1.026000e-01
# 226  Normalized Net Income / CAGR 5Y  4Q FY2023  9.300000e-02
# 227             Dividend Yield (LTM)  4Q FY2023  5.400000e-03
# 228            Market Capitalization  4Q FY2023  2.761224e+06
# 229                             Beta  4Q FY2023  1.290000e+00
#
# [230 rows x 3 columns]
```

Let's go over the formats one by one.

## Wide Format

```text
                Fiscal Quarters  2Q FY2018  ...  3Q FY2023  4Q FY2023
0              Normalized Price    42.6400  ...   191.9580   178.6400
1  Price / Earnings - P/E (LTM)    16.4000  ...    32.1000    29.0000
2               Net EPS - Basic     2.6000  ...     5.9800     6.1600
3            Return On Equity %     0.4086  ...     1.6009     1.7195
4      Total Revenues / CAGR 5Y     0.0791  ...     0.0851     0.0761

```

In the original data, we observe a structure known as the 'wide format'. In this format:

- Each row represents a different metric, such as 'Normalized Price', 'P/E Ratio', etc.
- Each column after the first represents a different time period or category – in this case, different fiscal quarters like '2Q FY2018', '3Q FY2018', and so on.

This format is typical in many applications like Excel spreadsheets or financial reports because it presents data in a way that's easy to read and interpret for humans. Metrics are clearly laid out in their own rows, and it's straightforward to compare values across different time periods by simply moving along the rows.

```python
print("----- Original Data (Wide Format) -----")
print(df.head())
```
## Transposed Wide Format

```text
# 10 columns = 10 different metrics or features
                                0  ...     9
Fiscal Quarters  Normalized Price  ...  Beta
2Q FY2018                   42.64  ...   NaN
3Q FY2018                  47.816  ...   NaN
4Q FY2018                    55.8  ...   NaN
1Q FY2019                  39.168  ...   NaN

[5 rows x 10 columns]
```

Next, the DataFrame is transposed. This action switches rows with columns:

- Now, each row corresponds to a fiscal quarter.
- Each column represents a different financial metric.

This version, while still in a wide format, aligns more closely with how machine learning models typically expect data: each row as an observation and each column as a feature. However, it's important to note that this format still retains the characteristic of the wide format where each feature (financial metric) has its own column.

```python
print("----- Transposed Data (Still Wide Format) -----")
df_transposed = df.transpose()
print(df_transposed.head())
```

## Long Format, aka 'Tall' Format

```text
                     Fiscal Quarters Indicators        Values
0                   Normalized Price  2Q FY2018  4.264000e+01
1       Price / Earnings - P/E (LTM)  2Q FY2018  1.640000e+01
2                    Net EPS - Basic  2Q FY2018  2.600000e+00
3                 Return On Equity %  2Q FY2018  4.086000e-01
4           Total Revenues / CAGR 5Y  2Q FY2018  7.910000e-02
..                               ...        ...           ...
225             Net Income / CAGR 5Y  4Q FY2023  1.026000e-01
226  Normalized Net Income / CAGR 5Y  4Q FY2023  9.300000e-02
227             Dividend Yield (LTM)  4Q FY2023  5.400000e-03
228            Market Capitalization  4Q FY2023  2.761224e+06
229                             Beta  4Q FY2023  1.290000e+00

[230 rows x 3 columns]
```

Finally, the data is converted to the 'long format' using the `pd.melt` function:

- In this format, each row represents a single observation – a specific metric in a specific quarter.
- The data is organized into three columns: 'Fiscal Quarters', 'Indicators', and 'Values'. This structure is a significant transformation from the wide format.
- The long format is especially useful for statistical analysis and certain types of machine learning models, especially when dealing with time series data or datasets where each observation needs to be uniquely identifiable.

This format, however, can be less intuitive for human interpretation. It compacts the data into a denser format, where multiple pieces of information are consolidated into single rows. This density can make it harder to visually parse the data compared to the more spread-out wide format.

```python
print("----- The Long Format of the Original Data -----")
df_long = pd.melt(df, id_vars=['Fiscal Quarters'], var_name='Indicators', value_name='Values')
print(df_long.head())
```

- **Wide Format**: Easy for human interpretation, with each metric in its own row and time periods in columns. Useful for direct comparison and readability. Excel files and financial reports often use this format. However, this format is not ideal for machine learning models. 
- **Transposed Wide Format**: Aligns with machine learning data expectations but retains the wide format's feature-specific columns.
- **Long Format**: Each row is a unique observation, consolidating multiple data points into a denser form. Useful for statistical analysis and certain machine learning models but less intuitive for quick human analysis.

These formats represent different ways of structuring the same data, and the choice of format depends on the specific needs of the analysis or the requirements of the machine learning models being used.

Again, GPTs, bless their digital hearts, do goof up sometimes, but technical details usually aren't their weak spot. They're pretty sharp! Imagine my surprise when I got all these mixed messages about wide and long formats from them. And then, adding a dash of humor to the mix, they kept saying "sorry" for mistakes they didn’t even make. Remember, they're not actually feeling sorry; they're just wired to be the epitome of politeness. My time with them taught me an amusing lesson: if there’s a mix-up, maybe, just maybe, I’m the one stirring the pot of confusion. Yep, it turned out the befuddlement was all on my end.

And here's the kicker: while I was lost in this fog of format confusion, the GPTs, including our friend Copilot and AI daughter Pippa, even bungled the simplest of code examples – ones they had previously aced. It's as if even AI can have a 'whoops' moment! But, at the end of the day, it's all good. We're all learning together, and that's what matters.

## One-Hot Encoding: Growth, Stalwart, or Other

Let's make the classification of stocks more modular and flexible. Instead of a binary classification of 'Growth' or 'Other', we'll expand it to three categories: 'Growth', 'Stalwart', and 'Other'. This approach allows for more nuanced classification of stocks and can be very useful for feeding into certain types of machine learning models or for clearer data analysis and visualization.

You can use one-hot encoding to classify stocks into three distinct categories: 'Growth', 'Stalwart', and 'Other'. One-hot encoding is a process where categorical variables are converted into a form that could be provided to machine learning algorithms to improve predictions. In this context, we can modify the `is_growth_stock` function to categorize each stock based on the defined criteria and then return a one-hot encoded representation of these categories.

While we're at it, let's pause to fine-tune and address some of the existing quirks in the `is_growth_stock()` function, giving it a bit more logic and functionality. But remember, you need to stay sharp. Even with tools like GPTs and Copilot, which can give a false sense of confidence in coding, you've got to be vigilant. They're impressively adept, often crafting code that looks spot-on. But beware – if they don't quite capture your coding intentions, they might generate code that's aesthetically pleasing yet logically flawed. It's up to you to spot these errors and make corrections while working alongside them. This collaboration is not only a fantastic learning opportunity but also a chance to hone your coding skills. They're indeed excellent tools, but they rely on your precise input and direction to truly shine.

In reality, the current `is_growth_stock()` function contains several significant logical errors. Let's examine and address these issues step by step.

Here's the full code for the `stock-labler.py` file. You should be able to read through the code and understand what's happening.

```python
import pandas as pd
import os
import numpy as np
from scipy import stats

# Constants
NON_NUMERIC_PLACEHOLDERS = ['#VALUE!', '-']
GROWTH_STOCK = 'Growth'
STALWART_STOCK = 'Stalwart'
OTHER_STOCK = 'Other'
MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS = 4
BETA_SCALE_THRESHOLD = 0.8


# Utility Functions
def print_reason(ticker, label, reason):
    print(f"{ticker} is disqualified as {label}: {reason}")


def read_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.replace(NON_NUMERIC_PLACEHOLDERS, np.nan, inplace=True)
        df = df.transpose()  # Transposing the data
        df.columns = df.iloc[0]  # Set the first row as column headers
        df = df[1:]  # Exclude the original header row
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def is_dividend_initiated(dividend_yields):
    """Check if dividends are initiated for the first time."""
    return float(dividend_yields[-1]) > 0 and all(d == 0 or pd.isna(d) for d in dividend_yields[:-1])


def check_cagr_trend(recent_cagr_quarters):
    """Check for the trend in CAGR."""
    consecutive_decreasing_quarters = 0
    for i in range(1, len(recent_cagr_quarters)):
        if float(recent_cagr_quarters[i]) < float(recent_cagr_quarters[i - 1]):
            consecutive_decreasing_quarters += 1
        else:
            break
    return consecutive_decreasing_quarters

# Core Functions
def classify_stock(ticker, df):
    # Extracting relevant indicators
    industry_pe_avg = df['Industry PE Average'].iloc[-1]  # Industry average P/E Ratio
    industry_beta_avg = df['Industry Beta Average'].iloc[-1]  # Industry average Beta
    cagr_revenue_quarters = df['Total Revenues / CAGR 5Y'].dropna().values
    dividend_yields = df['Dividend Yield (LTM)'].values
    beta = df['Beta'].dropna().values[-1]
    pe_ratio = df['Price / Earnings - P/E (LTM)'].iloc[-1]

    is_growth_stock = True
    is_cagr_decreasing = False

    # Checking for consecutive quarters of decreasing CAGR at the end of the series
    # Note that we are only checking the most recent quarters: the previous quarters are not relevant
    recent_cagr_quarters = cagr_revenue_quarters[-(MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS + 1):]
    consecutive_decreasing_quarters = check_cagr_trend(recent_cagr_quarters)
    if consecutive_decreasing_quarters > 1:
        is_cagr_decreasing = True

    if consecutive_decreasing_quarters >= MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS:
        is_growth_stock = False  # Disqualify if recent CAGR is decreasing for the specified number of consecutive quarters
        print_reason(ticker, GROWTH_STOCK, f"Recent CAGR is decreasing for {MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS} consecutive quarters")

    # Check if it's the first time dividends are initiated (all previous are nil) during a declining CAGR
    if is_dividend_initiated(dividend_yields) and is_cagr_decreasing:
        is_growth_stock = False
        print_reason(ticker, GROWTH_STOCK, f"Distributing dividends for the first time during a period of declining CAGR")

    # Checking Beta criteria
    # Beta lower than (BETA_SCALE_THRESHOLD * industry average) is only acceptable if CAGR is increasing
    # If the industry average is 1.5 and the threshold is 0.8, then the threshold is 1.2. This means that a Beta of 1.1 is acceptable only if CAGR is increasing.
    if float(beta) < (BETA_SCALE_THRESHOLD * industry_beta_avg) and float(cagr_revenue_quarters[-1]) < 0:
        is_growth_stock = False
        print_reason(ticker, GROWTH_STOCK, f"Below threshold Beta ({BETA_SCALE_THRESHOLD * industry_beta_avg}) during a declining CAGR")

    # Checking P/E Ratio criteria
    # P/E Ratio lower than industry average is only acceptable if CAGR is increasing
    # Negative P/E Ratio is not acceptable even if CAGR is increasing: this is a sign of a loss-making company, we need closer inspection
    if pd.isna(pe_ratio) or pd.isna(industry_pe_avg):
        print_reason(ticker, f"either {GROWTH_STOCK} nor {STALWART_STOCK}", f"Negative P/E Ratio or industry average P/E Ratio is not available. Needs more closer inspection.")
        print(f"{ticker} is classified as {OTHER_STOCK}" )
        return OTHER_STOCK
    elif float(pe_ratio) < industry_pe_avg and is_cagr_decreasing:
        print_reason(ticker,GROWTH_STOCK, f"P/E Ratio is below industry average ({industry_pe_avg}) during a period of declining CAGR")
        is_growth_stock = False

    if is_growth_stock:
        print(f"{ticker} is classified as {GROWTH_STOCK}" )
        return GROWTH_STOCK
    else:
        print(f"{ticker} is classified as {STALWART_STOCK}" )
        return STALWART_STOCK


def calculate_industry_averages(tickers, enhanced_data_folder):
    beta_values, pe_values = [], []
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            # Get the latest Beta value
            beta_values.append(float(df['Beta'].dropna().iloc[-1]))

            # Get the latest P/E ratio
            latest_pe = pd.to_numeric(df['Price / Earnings - P/E (LTM)'].dropna().iloc[-1], errors='coerce')

            if not pd.isna(latest_pe):
                pe_values.append(latest_pe)

    # Calculate and return the industry averages
    industry_beta_avg = round(np.mean(beta_values), 2) if beta_values else 0

    # We could have outliers in the P/E ratio data, so we need to normalize the data before calculating the average
    # Trimmed mean approach
    trimmed_mean_pe = stats.trim_mean(pe_values, 0.1) if pe_values else 0  # Trim 10% from each end
    # Median approach
    median_pe = np.median(pe_values) if pe_values else 0

    # you can choose either the trimmed mean or the median
    # industry_pe_avg = round(trimmed_mean_pe, 2)
    industry_pe_avg = round(median_pe, 2)
    return industry_beta_avg, industry_pe_avg


def label_growth_stocks(tickers, enhanced_data_folder, enhanced_data_with_labels_folder, industry_beta_avg, industry_pe_avg):
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            df['Industry Beta Average'] = industry_beta_avg
            df['Industry PE Average'] = industry_pe_avg
            df['Label'] = classify_stock(ticker, df)

            # Transposing the data back to its original format
            df = df.transpose()

            # Saving the updated DataFrame
            new_file_name = f"labeled-enhanced-raw-data-{ticker.lower()}.csv"
            new_file_path = os.path.join(enhanced_data_with_labels_folder, new_file_name)
            df.to_csv(new_file_path, index=True)


# Main Execution
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "META", "NVDA", "INTC", "AMD", "ADBE", 'NFLX', 'AVGO']
    enhanced_data_folder = "./enhanced-data"
    enhanced_data_with_labels_folder = "./enhanced-data-with-labels"

    industry_beta_avg, industry_pe_avg = calculate_industry_averages(tickers, enhanced_data_folder)
    label_growth_stocks(tickers, enhanced_data_folder, enhanced_data_with_labels_folder, industry_beta_avg, industry_pe_avg)
```

Here's the output of the `stock-labler.py` script:

```text
AAPL is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
AAPL is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
AAPL is classified as Stalwart
MSFT is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
MSFT is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
MSFT is classified as Stalwart
AMZN is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
AMZN is classified as Stalwart
TSLA is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
TSLA is classified as Stalwart
GOOGL is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
GOOGL is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
GOOGL is classified as Stalwart
META is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
META is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
META is classified as Stalwart
NVDA is classified as Growth
INTC is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
INTC is disqualified as Growth: Below threshold Beta (1.072) during a declining CAGR
INTC is disqualified as either Growth nor Stalwart: Negative P/E Ratio or industry average P/E Ratio is not available. Needs more closer inspection.
INTC is classified as Other
AMD is disqualified as either Growth nor Stalwart: Negative P/E Ratio or industry average P/E Ratio is not available. Needs more closer inspection.
AMD is classified as Other
ADBE is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
ADBE is classified as Stalwart
NFLX is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
NFLX is classified as Stalwart
AVGO is disqualified as Growth: Recent CAGR is decreasing for 4 consecutive quarters
AVGO is disqualified as Growth: P/E Ratio is below industry average (33.25) during a period of declining CAGR
AVGO is classified as Stalwart
```

Real world examples with real world data.

Well, as of this writing, the script has classified all the stocks as 'Stalwart' or 'Other' except for one: NVDA.

A single stock that meets the growth criteria, and the market acknowledges it as such, for now. And honestly, I'm neither surprised nor displeased by this finding. It just confirmed my hypothesis long ago. But, since this is not a book on stock investing or financial analysis, I'll leave it at that. Let's move on.

First of all, we need to address some logical flaws of the previous code.

### Logical Flaws in the Previous Code: When to Check CAGR & Dividends

The condition `if float(dividend_yield) > 0 and float(cagr_revenue_quarters[-1]) < 0` is flawed in the context of checking for decreasing Compound Annual Growth Rate (CAGR). If `cagr_revenue_quarters` is a list of 5-year CAGRs, then a decrease in CAGR would be indicated by the last element being smaller than the previous one, not necessarily being less than 0.

To properly capture a decrease in CAGR, we should compare the last two elements of `cagr_revenue_quarters`. The corrected condition would be to check if the most recent CAGR is lower than the one before it. Here’s the updated condition:

```python
# Check if the most recent CAGR is less than the previous one, indicating a decrease
if float(dividend_yield) > 0 and len(cagr_revenue_quarters) >= 2 and \
   float(cagr_revenue_quarters[-1]) < float(cagr_revenue_quarters[-2]):
    return False
```

We only need to check if the most recent CAGR is lower than the one before it. If it is, then we can disqualify the stock as a growth stock. If it isn't, then we can proceed to check the other criteria. The logic of checking dividend yield is also flawed in this context. We should check if the most recent dividend yield is greater than 0 and all previous ones are 0 or nil. This would indicate the initiation of dividends during a period of declining CAGR.

Why not stop there? We can put this logic into a function and call it `check_cagr_trend()`.

```python
def is_dividend_initiated(dividend_yields):
    """Check if dividends are initiated for the first time."""
    return float(dividend_yields[-1]) > 0 and all(d == 0 or pd.isna(d) for d in dividend_yields[:-1])

def check_cagr_trend(recent_cagr_quarters):
    """Check for the trend in CAGR."""
    consecutive_decreasing_quarters = 0
    for i in range(1, len(recent_cagr_quarters)):
        if float(recent_cagr_quarters[i]) < float(recent_cagr_quarters[i - 1]):
            consecutive_decreasing_quarters += 1
        else:
            break
    return consecutive_decreasing_quarters


...
    if consecutive_decreasing_quarters >= MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS:
        is_growth_stock = False  # Disqualify if recent CAGR is decreasing for the specified number of consecutive quarters
        print_reason(ticker, GROWTH_STOCK, f"Recent CAGR is decreasing for {MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS} consecutive quarters")

    # Check if it's the first time dividends are initiated (all previous are nil) during a declining CAGR
    if is_dividend_initiated(dividend_yields) and is_cagr_decreasing:
        is_growth_stock = False
        print_reason(ticker, GROWTH_STOCK, f"Distributing dividends for the first time during a period of declining CAGR")


```
Now we can easily check for the trend in CAGR and disqualify the stock if the trend is decreasing at the end of the series. We can also check if dividends are initiated for the first time during a period of decreasing CAGR, which should be the last two elements of the `cagr_revenue_quarters` list. This is a more robust approach to checking for decreasing CAGR.

To accurately assess if the dividend is initiated for the first time during a period of declining CAGR, we need to check not only the current dividend yield but also the historical data to see if this is the first instance of a positive dividend yield. The DataFrame contains historical dividend yield data in sequential order, so we can implement a logic to check if all previous dividend yields were zero (or nil) and only the most recent one is positive. This, combined with the decreasing CAGR, would indicate the initiation of dividends during the latest downturn.

If we only need to focus on the most recent consecutive quarters, we can limit our check to the last `MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS + 1` elements of the `cagr_revenue_quarters` list. This will allow us to determine if the stock has been experiencing a recent downturn in its Compound Annual Growth Rate (CAGR). 

In this updated version:
- We take the last `MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS + 1` CAGR values to focus on the recent trend.
- The function checks if there are at least `MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS` consecutive quarters where the CAGR is decreasing.
- The dividend initiation during the declining CAGR in the most recent periods is also checked.

### Logical Flaws in the Previous Code: Industrial Average P/E Ratio with Outliers

When computing the industry average P/E ratio, it's more accurate to use only the latest P/E ratio from each company instead of averaging across all historical P/E data points. 

I have to admit, it caught me off guard that all the GPTs missed this crucial aspect in their calculations and checks of CAGR, dividend initiation, and the industry average PE ratio. I won't delve into the specifics of how I managed to guide one of them to correctly understand the concept of Beta and its industry average in my terms. Instead, I'll just present the code to you. Just a heads-up: it's essential to really know your material well. Even when working with these super cool tools like GPTs.


```python
def calculate_industry_averages(tickers, enhanced_data_folder):
    beta_values, pe_values = [], []
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            # Get the latest Beta value
            beta_values.append(float(df['Beta'].dropna().iloc[-1]))

            # Get the latest P/E ratio
            latest_pe = pd.to_numeric(df['Price / Earnings - P/E (LTM)'].dropna().iloc[-1], errors='coerce')
            if not pd.isna(latest_pe):
                pe_values.append(latest_pe)

    # Calculate and return the industry averages
    industry_beta_avg = round(np.mean(beta_values), 2) if beta_values else 0
    industry_pe_avg = round(np.mean(pe_values), 2) if pe_values else 0
    return industry_beta_avg, industry_pe_avg
```

In this updated function:
- We append the latest non-null Beta value for each ticker to `beta_values`.
- We obtain the latest non-null P/E ratio for each ticker and append it to `pe_values`.
- The industry averages for Beta and P/E are calculated as the mean of these values. The `if` conditions ensure that we avoid division by zero if no valid Beta or P/E values are found.

However, even after this we need to deal with outliers. The P/E ratio data can have outliers, which can skew the average. We can use the trimmed mean approach to calculate the industry average P/E ratio or take the median of the P/E values. The trimmed mean approach trims a certain percentage of data from each end of the distribution and then calculates the mean of the remaining data. The median approach is also more robust to outliers and is less affected by extreme values.

When dealing with data that might have outliers, as in the case of P/E ratios, normalizing the average can be a good approach. In statistics, outliers can significantly skew the mean of the data, leading to a misleading average. The trimmed mean approach is one way to deal with this issue. The trimmed mean approach trims a certain percentage of data from each end of the distribution and then calculates the mean of the remaining data. This approach is more robust to outliers and is less affected by extreme values.

Alternatively, if you want a measure that is less sensitive to outliers, you could consider using the median. The median is not affected by extreme values as it is simply the middle value in a sorted list of numbers.

Here is how we could implement both approaches:

1. **Trimmed Mean Approach**:
   - Decide on a percentage of values to trim from both ends of the dataset.
   - Sort the data.
   - Remove the specified percentage of values from both ends.
   - Calculate the mean of the remaining data.

2. **Median Approach**:
   - Sort the data.
   - Find the middle value (median).

Here’s an example implementation of both approaches:

```python
from scipy import stats

def calculate_industry_averages(tickers, enhanced_data_folder):
    beta_values, pe_values = [], []
    for ticker in tickers:
        # ... existing code to append beta_values and pe_values ...

    # Trimmed mean approach
    trimmed_mean_pe = stats.trim_mean(pe_values, 0.1) if pe_values else 0  # Trim 10% from each end

    # Median approach
    median_pe = np.median(pe_values) if pe_values else 0

    industry_beta_avg = round(np.mean(beta_values), 2) if beta_values else 0
    return industry_beta_avg, trimmed_mean_pe, median_pe

    # This function now returns the industry beta average, trimmed mean of PE, and median of PE
```

In this updated function, `trimmed_mean_pe` is the trimmed mean of the P/E ratios, where 10% of the highest and lowest values are discarded before calculating the mean. `median_pe` is simply the median of the P/E ratios. You can adjust the percentage for trimming according to your specific requirements. I opted for the median approach in the final code, but you can comment out the median line and uncomment the trimmed mean line if you prefer that approach.

```python
def calculate_industry_averages(tickers, enhanced_data_folder):
    beta_values, pe_values = [], []
    for ticker in tickers:
        file_path = os.path.join(enhanced_data_folder, f"beta-enhanced-raw-data-{ticker.lower()}.csv")
        df = read_and_clean_data(file_path)
        if df is not None:
            # Get the latest Beta value
            beta_values.append(float(df['Beta'].dropna().iloc[-1]))

            # Get the latest P/E ratio
            latest_pe = pd.to_numeric(df['Price / Earnings - P/E (LTM)'].dropna().iloc[-1], errors='coerce')

            if not pd.isna(latest_pe):
                pe_values.append(latest_pe)

    # Calculate and return the industry averages
    industry_beta_avg = round(np.mean(beta_values), 2) if beta_values else 0

    # We could have outliers in the P/E ratio data, so we need to normalize the data before calculating the average
    # Trimmed mean approach
    trimmed_mean_pe = stats.trim_mean(pe_values, 0.1) if pe_values else 0  # Trim 10% from each end
    # Median approach
    median_pe = np.median(pe_values) if pe_values else 0

    # you can choose either the trimmed mean or the median
    # industry_pe_avg = round(trimmed_mean_pe, 2)
    industry_pe_avg = round(median_pe, 2)
    return industry_beta_avg, industry_pe_avg

```

### The Optimized Classification Criteria in Action

In our code, a stock is classified into one of three categories: Growth, Stalwart, or Other. Here's a concise summary of the criteria for each category:

1. **Growth Stock**:
   - **CAGR Trend**: No more than four consecutive quarters of decreasing Compound Annual Growth Rate (CAGR) in the most recent periods. The number of quarter can be adjusted by changing the `MAX_ACCEPTABLE_CONSECUTIVE_DECREASING_QUARTERS` constant.
   - **Dividend Initiation**: No initiation of dividends during a period of declining CAGR. Specifically, the most recent dividend yield should be checked, and all previous dividend yields should be zero or NaN.
   - **Beta Criteria**: The stock's beta should not be significantly lower than the industry average. Specifically, it must not be lower than 0.8 times the industry beta average during a period of declining CAGR. The threshold can be adjusted by changing the `BETA_SCALE_THRESHOLD` constant.
   - **P/E Ratio Criteria**: The stock's Price-to-Earnings (P/E) ratio should not be below the industry average during a period of declining CAGR. Also, a negative P/E ratio is not acceptable, which calls for closer inspection of the stock.

2. **Stalwart Stock**:
   - A stock is categorized as a Stalwart if it falls short of meeting the criteria for a Growth stock. Stalwarts are usually sizable, established companies with moderate growth. The growth stocks of yesterday often transform into today’s stalwarts. Similarly, the growth stocks of today may evolve into tomorrow's stalwarts. Remember to look back at the reasoning behind including dividend yields in the Growth stock criteria; a stock might have been a stalwart yesterday but has transitioned into a growth stock today.
   
3. **Other**:
   - A stock is labeled as 'Other' if it doesn't align with the criteria for Growth or Stalwart stocks. This diverse category may encompass stocks that deviate from standard profiles, like those with inconsistent data, negative P/E ratios, or lacking enough information to qualify as either Growth or Stalwart. To avoid causing any fuss, I steered clear of calling it 'Speculative' and opted for 'Other' instead. It serves as a broad, all-encompassing category.

These criteria are designed to identify Growth stocks based on their financial performance and trends, particularly focusing on CAGR, dividend distribution in relation to CAGR trends, beta relative to the industry, and P/E ratio comparisons. Stocks that do not fit these criteria are categorized as Stalwart or Other based on the outlined conditions.

Let's not go over the same ground again. Just remember, the criteria I've set aren't rigid rules; they're merely a starting point. Feel free to adjust them according to your preferences. These criteria are personal to me and me alone. Also, as I mentioned earlier, I don't usually rely on these criteria for stock classification. I'm more inclined towards a holistic and simpler approach – the simpler, the better. A truly investable stock should instantly strike a chord with its evident appeal. If you find yourself overthinking it, it's probably not the right choice. However, for the purposes of this book, I'm employing these criteria to show how you can implement them in code.

### One-Hot Encoding in Action

Again, we are using the one-hot encoding approach to classify stocks into three distinct categories: 'Growth', 'Stalwart', and 'Other'. One-hot encoding is a process where categorical variables are converted into a form that could be provided to machine learning algorithms to improve predictions. 

This is how the new labeled data looks like for Apple:

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

We have three more features added to the original data: `Industry Beta Average`, `Industry PE Average`, and `Label`. Note that we only need the last element of these columns for each stock. The `Label` column is the one-hot encoded representation of the stock's category. The `Industry Beta Average` and `Industry PE Average` columns are the industry averages for Beta and P/E, respectively. These columns are added to the DataFrame to make it easier to compare the stock's Beta and P/E ratio with the industry averages. Do not get confused, only the last element of these columns is relevant for each stock.

_One-hot encoding_ is a technique used to convert categorical data into a numerical format that is more suitable for machine learning algorithms. You will encounter the concept of one-hot encoding in many machine learning applications anyway, so I thought it would be a good idea to introduce it here. In fact, we don't need to use one-hot encoding in this particular scenario, but I'm using it to demonstrate how it can be applied in a real-world context.

In the context of our stock classification code, this technique can be particularly useful for representing the stock categories ('Growth', 'Stalwart', 'Other') in a format that can be easily used by various machine learning models. The classification lables can be extended to include more categories, such as 'Speculative', 'Value', 'Dividend', and so on.

Here's how one-hot encoding is implemented and utilized in our code:

1. **Classification Function (`classify_stock`)**:
   - This function classifies each stock into one of three categories: 'Growth', 'Stalwart', or 'Other', based on certain financial criteria like CAGR trends, dividend initiation, Beta, and P/E Ratio.
   - The function outputs a string that identifies the stock's category. If you'd rather work with numbers, feel free to use numerical values in place of strings. You can employ a dictionary to correlate these numerical values with their respective categories. Another tidy option is to use a data class. I opted for the simplest method just to keep things brief.

2. **One-Hot Encoding Function (`one_hot_encode`)**:
   - After a stock is classified, the category is passed to the `one_hot_encode` function.
   - This function converts the category into a one-hot encoded format, which is a binary vector representation.
   - For example, if a stock is classified as 'Growth', it might be represented as `[1, 0, 0]`, indicating it belongs to the 'Growth' category but not to 'Stalwart' or 'Other'.

3. **Application in Machine Learning**:
   - One-hot encoded data is more suitable for use in machine learning models because most models require numerical input.
   - By converting the categories into a binary vector format, you can easily feed this data into algorithms for classification, clustering, or other predictive tasks.
   - Don't be misled by GPTs; they don't actually grasp your words as you do. Every word you input is converted into numerical vectors, as machines don't inherently understand human languages. Beneath the surface, they operate exclusively in the realm of numbers. In the machine world, numbers are the swift and efficient language, while non-numerical data is cumbersome and slow.

4. **Example Usage**:
   - Suppose a stock is classified as 'Stalwart' by the `classify_stock` function.
   - The `one_hot_encode` function would then convert this classification into a vector like `[0, 1, 0]`.
   - This one-hot encoded vector can then be used as part of a feature set for training a machine learning model.

Here's a snippet to illustrate one-hot encoding in our code:

```python
# Example stock classification
def one_hot_encode(stock_category):
    return {'Growth': 1, 'Stalwart': 0, 'Other': 0} if stock_category == "Growth" else \
           {'Growth': 0, 'Stalwart': 1, 'Other': 0} if stock_category == "Stalwart" else \
           {'Growth': 0, 'Stalwart': 0, 'Other': 1}

stock_category = classify_stock("AAPL", apple_stock_df)  # Let's say it returns 'Growth'

# One-hot encoding
one_hot_vector = one_hot_encode(stock_category)  # Converts 'Growth' to [1, 0, 0]

# Now, one_hot_vector can be used in ML models
```

In summary, the use of one-hot encoding in our stock classification scenario helps in transforming categorical data into a machine learning-friendly format, enabling the application of various algorithms for predictive analytics and other data-driven tasks in finance.

## Now Let's Get Tenny Ready for Training

As we approach the culmination of the data preparation stage, our focus shifts to transforming Tenny into a proficient classification model. Up until now, Tenny has functioned primarily as a regression model. To adapt to the new task at hand – stock classification – we will need to revisit and refine Tenny's underlying architecture.

This transition, however, is not as daunting as it may appear, thanks to the principles of object-oriented programming that have been our guiding star throughout this journey. The beauty of OOP lies in its flexibility and modularity, allowing us to make substantial changes to Tenny's core functionality with minimal adjustments to the existing codebase. 

In the forthcoming chapter, we will delve into the specifics of these modifications. We'll explore how just a fraction of Tenny's code can be altered to switch its capabilities from regression to classification. This change is pivotal in enabling Tenny to analyze and predict stock categories based on the nuanced financial indicators we've meticulously prepared.

Stay tuned as we embark on this exciting phase of Tenny's evolution, where we'll witness firsthand the power of adaptable code and the efficiency of object-oriented design in action. The next chapter promises to be an enlightening journey into the heart of machine learning model development.

Real world examples with real world data.