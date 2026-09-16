# Best programs per arm-seed (train-F1 incumbent); both modules shown, [module] headers

## bo seed 0 (train F1 0.758, held F1 0.692, held recall 0.835; lineage ['answerer'])

```
[selector]
Below are ten Wikipedia paragraphs, each starting with its title, followed by a question. Select the paragraphs that contain the facts needed to answer the question - usually two: one about the entity the question names and one about the entity it leads to. Return their titles exactly as written.

Paragraphs:
{context}

Question: {question}

[answerer]
Use the paragraphs below to answer the question. Provide the shortest exact answer that directly matches the information given, including phrases, names, dates, or numbers from the text. For yes/no questions, respond with 'yes' or 'no'.

Paragraphs:
{context}

Question: {question}
```

### recombination (existing=False; train F1 0.665, held F1 0.678)

```
[selector]
Below are ten Wikipedia paragraphs, each beginning with a title, followed by a question. Identify and list the titles of the two paragraphs that are most relevant to answering the question. Your selection should include:
1. The paragraph about the entity named in the question.
2. The paragraph about the entity the question leads to.

Paragraphs:
{context}

Question: {question}

[answerer]
Extract the exact answer from the context that precisely matches the reference answer for the given question. Focus on selecting the shortest span that achieves the highest F1 score.

Context:
{context}

Question: {question}

Exact Answer:
```

## bo seed 1 (train F1 0.744, held F1 0.721, held recall 0.873; lineage ['selector', 'selector'])

```
[selector]
Select, from the provided Wikipedia paragraphs, the ones needed to answer the question - usually two: one about the entity the question names and one about the entity it leads to. Return their titles exactly as written. Ensure that both entities involved in the question are represented among the selected paragraphs.

Paragraphs:
{context}

Question: {question}

Additional Instructions:
1. Double-check to ensure both entities involved in the multi-hop question are included in your selection.
2. If a direct mention of an entity in the question is not found in the context, look for closely related information that indirectly answers the question.
3. Prioritize accuracy over additional information; do not include paragraphs that do not contribute to answering the question.

[answerer]
Answer the question using the paragraphs below. Give the shortest exact answer: a name, date, number or phrase copied from the text, or yes / no for yes-no questions.

Paragraphs:
{context}

Question: {question}
```

### recombination (existing=True; train F1 0.744, held F1 0.721)

```
[selector]
Select, from the provided Wikipedia paragraphs, the ones needed to answer the question - usually two: one about the entity the question names and one about the entity it leads to. Return their titles exactly as written. Ensure that both entities involved in the question are represented among the selected paragraphs.

Paragraphs:
{context}

Question: {question}

Additional Instructions:
1. Double-check to ensure both entities involved in the multi-hop question are included in your selection.
2. If a direct mention of an entity in the question is not found in the context, look for closely related information that indirectly answers the question.
3. Prioritize accuracy over additional information; do not include paragraphs that do not contribute to answering the question.

[answerer]
Answer the question using the paragraphs below. Give the shortest exact answer: a name, date, number or phrase copied from the text, or yes / no for yes-no questions.

Paragraphs:
{context}

Question: {question}
```

## bo seed 2 (train F1 0.710, held F1 0.720, held recall 0.882; lineage ['selector'])

```
[selector]
Given ten Wikipedia paragraphs and a question, identify and select the two paragraphs that are most relevant to answering the question. Typically, you need the paragraph about the question's subject and another paragraph about the entity it leads to. Ensure that your selections directly address the entities mentioned in the question. List the titles of these paragraphs exactly as provided.

Paragraphs:
{context}

Question: {question}

[answerer]
Answer the question using the paragraphs below. Give the shortest exact answer: a name, date, number or phrase copied from the text, or yes / no for yes-no questions.

Paragraphs:
{context}

Question: {question}
```

### recombination (existing=False; train F1 0.648, held F1 0.701)

```
[selector]
Identify the two key Wikipedia paragraphs needed to answer the question. These should be the one that covers the main entity in the question and another that addresses the related entity. Look through the given paragraphs and select their titles precisely as they are listed.

Paragraphs:
{context}

Question: {question}

[answerer]
Using the information from the paragraphs below, provide the shortest and most accurate answer by copying the exact text (name, date, number, phrase) or responding with 'yes' or 'no' for yes-no questions.

Paragraphs:
{context}

Question: {question}
```

## gepa seed 0 (train F1 0.762, held F1 0.698, held recall 0.848; lineage ['selector', 'answerer'])

```
[selector]
Below are ten Wikipedia paragraphs, each starting with its title, followed by a question. Select the paragraphs that contain the facts needed to answer the question - usually two: one about the entity the question names and one about the entity it leads to. Return their titles exactly as written. Additionally, include any intermediate entity paragraphs needed for a multi-hop question. Make sure to select paragraphs that directly contribute to answering the question.

Paragraphs:
{context}

Question: {question}

[answerer]
Answer the question using the paragraphs below. Give the shortest exact answer: a name, date, number, phrase copied directly from the text, or a simple word like yes / no.

Paragraphs:
{context}

Question: {question}
```

## gepa seed 1 (train F1 0.717, held F1 0.676, held recall 0.820; lineage ['selector'])

```
[selector]
Below are ten Wikipedia paragraphs, each starting with its title, followed by a question. To answer the multi-hop question, select the paragraphs that provide the necessary facts - typically two: one about the entity the question mentions and another about the entity it leads to. Return the titles of these paragraphs exactly as written, ensuring to include any additional relevant paragraphs that directly contribute to answering the question.

Paragraphs:
{context}

Question: {question}

[answerer]
Answer the question using the paragraphs below. Give the shortest exact answer: a name, date, number or phrase copied from the text, or yes / no for yes-no questions.

Paragraphs:
{context}

Question: {question}
```

## gepa seed 2 (train F1 0.668, held F1 0.737, held recall 0.847; lineage ['selector'])

```
[selector]
Below are ten Wikipedia paragraphs, each starting with its title, followed by a question. Select the paragraphs that contain the facts needed to answer the question - usually two: one about the entity the question names and one about the entity it leads to. Return their titles exactly as written. Select paragraphs that are directly relevant to the question. If there is any ambiguity in the question, choose the most straightforward interpretation. Make sure to include all necessary paragraphs to fully answer the question.

Paragraphs:
{context}

Question: {question}

[answerer]
Answer the question using the paragraphs below. Give the shortest exact answer: a name, date, number or phrase copied from the text, or yes / no for yes-no questions.

Paragraphs:
{context}

Question: {question}
```
