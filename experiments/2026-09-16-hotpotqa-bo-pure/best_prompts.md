# Best selector per seed (train-F1 incumbent), bo-pure arm

## seed 0 (train F1 0.762, held F1 0.727, held recall 0.872, depth 5)

```
Your task is to analyze the given Wikipedia paragraphs and question, and select the two most relevant paragraphs to answer the question. These should be:

1. The first paragraph about the main entity
2. The second paragraph about the secondary entity

Consider these instructions:

- Choose paragraphs that are directly relevant to the main and secondary entities in the question.
- If a paragraph discusses both entities, use it for the main entity and find another for the secondary entity.
- Focus on paragraphs that directly address the question.

Here are the paragraphs:
{context}

And the question is: {question}

Return the titles of the two selected paragraphs in the specified order.
```

## seed 1 (train F1 0.731, held F1 0.694, held recall 0.838, depth 3)

```
You need to select two relevant Wikipedia paragraphs to answer the given question.

Guidelines:

1. Choose one paragraph that directly mentions the subject of the question.
2. Select another paragraph that either connects to a related entity or adds the necessary context.
3. If the directly mentioned paragraph provides a sufficient standalone answer, ensure the second paragraph supplements this.
4. Ensure you return the titles of the two selected paragraphs only.
5. Keep any placeholders or specific wording from the question intact.

Context paragraphs:
{context}

Question: {question}
```

## seed 2 (train F1 0.702, held F1 0.746, held recall 0.885, depth 6)

```
Given the following ten Wikipedia paragraphs, identify the two that are most relevant to answering the given question. These paragraphs should include both the primary entity and the related entity, and clearly demonstrate their connection.
Paragraphs:
{context}

Question: {question}

Guidelines:
- Select paragraphs where both entities are mentioned in a clear and relevant context.
- Ensure the paragraphs explicitly describe a relationship or connection between the entities.
- The two selected paragraphs should be those that provide the most direct and relevant information to answer the question.
- List the titles of the selected paragraphs in the order they appear in the paragraph list.
- If the direct connection is not found within a paragraph, consider the context of surrounding paragraphs to understand the relationship.
- Prioritize paragraphs that directly address the entities' relationship over those that only mention the entities in passing.
- If one paragraph does not cover both entities, look for a second paragraph that complements the first by providing the missing information.
- Ensure that the selected paragraphs form a coherent narrative that answers the question without needing additional context.
```
