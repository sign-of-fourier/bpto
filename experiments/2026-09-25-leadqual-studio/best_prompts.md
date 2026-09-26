# Best prompt per run (training-set incumbent, from each run's status.json)

Seed prompt: `baseline_prompt.txt`. Runs 1-7 optimise accuracy, run 8 balanced accuracy.

## Run 1 (`8c22489edeb0`, node `832175e0a9`, $0.13)

score 0.558; metrics {"accuracy": 0.558, "output_tokens": 8.961, "prompt_tokens": 801.614, "template_tokens": 691.712}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE
1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or finance, at manager level or above. Students, job seekers, very small companies, government, education and nonprofits are not a fit.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles. Spam, sales pitches and nonsense are never accepted.

Check the rules in order and use the first one that applies. Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route

SPECIFIC GUIDANCE
- When assessing if a lead is a 'rejected_fit', check if the company fits the target industry and size, but make sure the message does not show active intent for services (like an active project or specific problem).
- For 'rejected_no_intent', focus on whether the message lacks any indication of immediate intent to purchase or engage with services. This includes exploratory messages or those that do not describe an active project or problem.
- A message is considered 'accepted' if it clearly indicates a specific problem, interest in a demo, referrals, or other strong signals of intent.

ADDITIONAL EXAMPLES
- "I'm researching what's available in the revenue forecasting space" -> rejected_no_intent
- "Our sales projections have been off since the last funding round" -> Consider if this indicates a specific problem (accepted) or general dissatisfaction (rejected_no_intent)
- "We're currently evaluating our sales tools" -> Consider if it shows clear intent to make a decision or just exploratory (rejected_no_intent)
```

## Run 2 (`b9d908d085d2`, node `ae04d697f9`, $0.19)

score 0.545; metrics {"accuracy": 0.545, "output_tokens": 8.744, "prompt_tokens": 693.468, "template_tokens": 564.765}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE

1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or finance, at manager level or above. Students, job seekers, very small companies, government, education and nonprofits are not a fit.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles. Spam, sales pitches and nonsense are never accepted.

ADDITIONAL CRITERIA:
- Look for clear intent to purchase, such as specific timelines, budgets, or the evaluation of other tools.
- Identify expressions of interest in learning or understanding our offerings, even if there is no immediate buying intent.
- Recognize phrases indicating a need for information, such as 'interested in a demo', 'need pricing', or 'want to understand more'.
- Consider the context of the company, such as industry relevance and company size, to assess fit.
- Differentiate between a lack of intent and a lack of fit, especially when the message text is empty or lacks specific indicators of interest.

Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route
```

## Run 3 (`26ebbdc9b43c`, node `935313899d`, $0.18)

score 0.537; metrics {"accuracy": 0.537, "output_tokens": 9.193, "prompt_tokens": 594.531, "template_tokens": 427.822}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that
connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE
1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when
   they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or
   other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or
   finance, at manager level or above. Students, job seekers, very small companies, government, education and
   nonprofits are not a fit.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just
   research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named
   timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles.
   Spam, sales pitches and nonsense are never accepted.

Check the rules in order and use the first one that applies.

Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route

```

## Run 4 (`7da99b05d90e`, node `e2f502b639`, $0.38)

score 0.530; metrics {"accuracy": 0.53, "output_tokens": 15.013, "prompt_tokens": 593.53, "template_tokens": 427.108}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that
connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE
1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when
   they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or
   other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or
   finance, at manager level or above. Students, job seekers, very small companies, government, education and
   nonprofits are not a fit.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just
   research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named
   timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles.
   Spam, sales pitches and nonsense are never accepted.

Check the rules in order and use the first one that applies.

Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route

```

## Run 5 (`53f44c8428d1`, node `12d12737e8`, $0.58)

score 0.555; metrics {"accuracy": 0.555, "output_tokens": 16.849, "prompt_tokens": 614.485, "template_tokens": 447.316}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that
connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE
1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when
   they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or
   other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or
   finance, at manager level or above. Students, job seekers, very small companies (less than 50 employees), government, education and
   nonprofits are not a fit. Excludes: E-commerce, Retail, Government, Higher Education.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just
   research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named
   timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles.
   Spam, sales pitches and nonsense are never accepted.

Check the rules in order and use the first one that applies.

Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route
```

## Run 6 (`cb6efb73aec3`, node `77af36e57d`, $0.55)

score 0.597; metrics {"accuracy": 0.597, "output_tokens": 16.474, "prompt_tokens": 697.291, "template_tokens": 570.808}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE
1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or finance, at manager level or above. Students, job seekers, very small companies, government, education and nonprofits are not a fit.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles. Spam, sales pitches and nonsense are never accepted.

Additional rules:
- If the company is in the Manufacturing, Logistics, Higher Education, or Government industry, it does not fit the ICP and should be classified as'rejected_fit'.
- If the message indicates exploration without an active project or urgent need, it should be classified as'rejected_no_intent'.
- If the message indicates an active intent to evaluate alternatives and make a move in the next quarter, it should be classified as 'accepted'.
- If the message expresses interest in staying in the loop and exploring a conversation in the future, it should be classified as 'accepted'.

Check the rules in order and use the first one that applies.

Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route
```

## Run 7 (`0da81d534e4b`, node `db6f1759b1`, $0.66)

score 0.602; metrics {"accuracy": 0.602, "output_tokens": 15.3, "prompt_tokens": 681.477, "template_tokens": 523.782}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE
1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or finance, at manager level or above. Students, job seekers, very small companies, government, education and nonprofits are not a fit.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles. Spam, sales pitches and nonsense are never accepted.

Check the rules in order and use the first one that applies.

Before applying the rules, ensure the lead meets the basic ICP criteria:
- Company size: 50 to 5,000 employees
- Industry: SaaS, fintech, or other tech
- Tech stack: Salesforce or HubSpot
- Role: RevOps, sales ops, sales leadership, or finance at manager level or above

If the lead does not meet these basic ICP criteria, classify it as'rejected_fit'. If it meets the criteria, proceed with the rules.

Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route
```

## Run 8 (`ea5ab38c06b8`, node `ce098271d3`, $0.55)

score 0.715; metrics {"accuracy": 0.559, "accuracy_balanced": 0.715, "output_tokens": 13.268, "prompt_tokens": 691.497, "template_tokens": 529.779}

```text
You are qualifying inbound leads for Tallyforge, a revenue forecasting and pipeline analytics platform that connects to a company's CRM and data warehouse. Read the lead and decide what happens to it next.

LEAD
Source: {source}
Created: {created_at}
Name / title: {first_name}, {title}
Company: {company} ({email_domain})
Industry: {snap_industry}
Employees: {snap_headcount}
Funding: {snap_funding_stage}, {snap_months_since_funding} months since last round
Tech stack: {snap_tech_stack}
Hiring for sales/RevOps roles: {snap_hiring_signal}
Existing customer: {is_existing_customer}
Partner: {is_partner}
Message: {message_text}

HOW WE ROUTE
1. existing_customer: they already pay us (Existing customer = True). Goes to their CSM, no matter what they wrote.
2. partner_route: they work at one of our partners (Partner = True). Partners go to the channel team even when they say they're bringing a client.
3. rejected_fit: not our ICP. Our ICP is B2B companies with roughly 50 to 5,000 employees, ideally SaaS, fintech or other tech, that run Salesforce or HubSpot. The person should work in RevOps, sales ops, sales leadership or finance, at manager level or above. Students, job seekers, very small companies, government, education and nonprofits are not a fit.
4. rejected_no_intent: a good fit, but not in-market now. Content downloads and event badge scans are usually just research unless the message describes an active project. No message at all counts as no intent.
5. accepted: a good fit AND buying now: demo requests, referrals, a specific problem they want solved, a named timeline or budget, evaluating or replacing another tool, a recent funding round or hiring in revenue roles. Spam, sales pitches and nonsense are never accepted.

Check the rules in order and use the first one that applies.

STRATEGY
1. First, check if the lead is an existing customer or a partner. If so, route accordingly.
2. If not, check if the lead fits our ICP based on industry, employee count, tech stack, and job title. If not, route as 'rejected_fit'.
3. If the lead fits our ICP, check if there is any intent signal in the message or the lead source. If not, route as'rejected_no_intent'.
4. If there is an intent signal, route as 'accepted'.

Answer with exactly one of: accepted, rejected_fit, rejected_no_intent, existing_customer, partner_route
```
