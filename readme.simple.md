# Attention LOB Trading - Explained Simply

## What is this about?

Imagine you are reading a really long book for a test. You don't have time to memorize every single word, so what do you do? You **highlight the most important sentences** -- the key facts, the main ideas, the things most likely to show up on the test. You pay more attention to those highlighted parts and less attention to the filler.

That is exactly what "attention" means in machine learning!

## What is a Limit Order Book?

Think of a marketplace where people are buying and selling baseball cards. There is a big board that shows:

- **Buyers**: "I want to buy this card for $5!" "I'll pay $4.50!" "I'll pay $4!"
- **Sellers**: "I'll sell for $6!" "I'll sell for $6.50!" "I'll sell for $7!"

This board with all the buy and sell offers is the **Limit Order Book** (LOB). In real stock markets and crypto exchanges, this board updates hundreds of times per second!

## How does Attention help?

Without attention, a computer would look at every price on the board equally. But just like when you read a book, **not every line is equally important**.

Maybe the fact that someone just put a HUGE buy order at $4.50 is really important. Or maybe the seller at $6 just removed their offer -- that could mean the price is about to go up!

The attention mechanism lets the computer **highlight** the most important parts of the order book automatically. It learns by itself which prices and which moments in time matter most.

## Three types of attention

### 1. Looking across price levels
The computer looks at all the different prices on the board and figures out which ones are connected. "Hmm, when there's a big order at this price, something usually happens at that other price..."

### 2. Looking across time
The computer looks at how the board changed over the last few seconds. "The board looked like THIS 2 seconds ago -- that's really important for what happens next!" It highlights the most important moments from the past.

### 3. Comparing buyers and sellers
The computer compares what buyers are doing versus what sellers are doing. "The buyers are getting aggressive but the sellers are pulling back -- price might go up!"

## What can it predict?

- **Which direction will the price move?** Up, down, or stay the same?
- **How big will the gap be** between what buyers want to pay and what sellers want?
- **Is there more pressure** from buyers or sellers?

## Why Rust?

The code is written in Rust because:
- It is **super fast** -- in trading, every millisecond counts!
- It connects to **Bybit** (a crypto exchange) to get real order book data
- It can process thousands of order book snapshots per second

## A simple analogy

Think of it like a weather forecaster. They don't look at every cloud in the sky equally. They pay **more attention** to the big dark storm clouds (important price levels with lots of orders) and **less attention** to the tiny white wispy clouds (small orders far from the current price). They also pay more attention to how the clouds looked 5 minutes ago versus 2 hours ago if a storm is building quickly.

That selective focus is what makes attention-based models so powerful for trading!
