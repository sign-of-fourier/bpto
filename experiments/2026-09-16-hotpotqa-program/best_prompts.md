## bo seed 0 (train F1 0.761, held F1 0.697, held recall 0.843)

```
Given ten Wikipedia paragraphs, each with a title followed by a question, your task is to pick the two paragraphs essential for answering the question. These paragraphs should be about the entity named in the question and another entity it connects to. If the question involves a person, include the paragraph detailing their birth or death if relevant. Return the titles of the selected paragraphs exactly as they appear.

Paragraphs:
{context}

Question: {question}
```

## bo seed 1 (train F1 0.716, held F1 0.683, held recall 0.815)

```
To answer the question, select the relevant paragraphs from the following list of ten Wikipedia paragraphs, each identified by its title. You are likely to need two paragraphs, one for the question's primary subject and one for its related subject.

- **Prioritize selecting paragraphs that directly talk about both the main subject and any related subjects of the question.**
- **Make sure to consider paragraphs that mention key events, relationships, or details pertinent to answering the question.**
- **Do not include paragraphs that only provide general information not directly related to the question.**
- **Return the titles of the selected paragraphs, preserving the order as they are listed.**

Paragraphs:
{context}

Question: {question}
```

## bo seed 2 (train F1 0.672, held F1 0.705, held recall 0.815)

```
Given a list of ten Wikipedia paragraphs, each paragraph title followed by a question, select the paragraphs that are essential to answer the question correctly. Typically, this involves selecting two paragraphs: one detailing the main entity and another detailing an entity it relates to. Return the titles of these paragraphs exactly as they appear. In cases where a question requires deeper context, select any additional paragraphs necessary to provide a comprehensive answer.

Paragraphs:
{context}

Question: {question}
```

## gepa seed 0 (train F1 0.746, held F1 0.740, held recall 0.895)

```
Below are ten Wikipedia paragraphs, each starting with its title, followed by a question. Select the paragraphs that contain the facts needed to answer the question - usually two: one about the entity the question names and one about the entity it leads to. Return their titles exactly as written.

Paragraphs:
{context}

Question: {question}


 
   Follow these rules:

1. Select the paragraph that gives you information directly about the subject of the question. This is usually the first paragraph in the set.
2. Identify the entity that the subject of the question leads to, and select the paragraph that provides relevant information about this secondary entity.
3. If the question asks for a comparison between two entities, select paragraphs for both entities.
4. Do not select paragraphs that provide unrelated information.
5. Ensure the order of your selection matches the order in which the information appears in the paragraphs.

For example, if the question is about Damian Green and e-Borders, the order should be 'Damian Green' followed by 'E-Borders'.
```

## gepa seed 1 (train F1 0.736, held F1 0.674, held recall 0.782)

```
To address the given question, carefully select two Wikipedia paragraphs that provide the necessary facts to answer it. The chosen paragraphs must specifically contain information about the main entity mentioned in the question and the entity it leads to. Ensure to return their titles exactly as they are written.

Paragraphs:
{context}

Question: {question}
```

## gepa seed 2 (train F1 0.699, held F1 0.742, held recall 0.922)

```
Below are ten Wikipedia paragraphs, each starting with its title, followed by a question. Select the paragraphs that contain the facts needed to answer the question - usually two: one about the entity the question names and one about the entity the question leads to. To ensure the most accurate answer, consider the following guidelines:

- **Identify the Key Entities:** Select the paragraph that provides the most relevant information about the primary entity the question names.
- **Identify the Target Entity:** Select the paragraph that provides the most relevant information about the secondary entity the question leads to.
- **Evaluate Relevance:** If multiple paragraphs provide equally relevant information about an entity, choose the one that provides the most comprehensive context.
- **Ensure Completeness:** Ensure that the selected paragraphs together provide enough information to fully answer the question.
- **Avoid Redundancy:** Avoid selecting paragraphs that provide overlapping information without adding new insights.
- **Follow the Exact Titles:** Return the titles of the selected paragraphs exactly as written.
- **Return Empty if None:** If no relevant paragraphs are found, return an empty list.

**Additional Guidelines:**

- **Check All Paragraphs:** If the answer depends on two distinct entities or facts, make sure both are covered by your selection.
- **Prioritize Specificity:** When a paragraph seems relevant, prioritize those that contain specific details over those that provide broader context.
- **Confirm Context Relevance:** Ensure the selected paragraphs not only mention the entities but also provide the necessary context to answer the question.
- **Watch for Ambiguous Terms:** Be cautious with paragraphs that use terms which could refer to multiple entities, and clarify by checking the question context.

**Paragraphs:**
{context}

**Question:** {question}

When selecting the paragraphs, follow this order:
1. Look for the paragraph that directly mentions the primary entity.
2. Look for the paragraph that provides the necessary context for the secondary entity.
3. Ensure both paragraphs together provide a complete answer.

For example, if the question is about a specific event in a person's life, first find the paragraph that mentions the person, then find the paragraph that provides the details of the event.

Finally, check that the selected paragraphs do not contain redundant information and that the question can be answered comprehensively using them.
```

