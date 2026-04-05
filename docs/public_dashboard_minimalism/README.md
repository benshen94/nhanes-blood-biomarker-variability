# Public Dashboard Lessons

## Goal

This note captures the design corrections from the audience-facing aging biomarkers dashboard pass.
The main lesson is simple: a public exploration page should feel calm before it feels comprehensive.

## What We Changed

- We removed always-visible helper panels that competed with the charts.
- We moved secondary explanation into click-to-open bubbles.
- We shortened the hero so the page opens with one idea instead of several.
- We removed optional comparison clutter that was not central to the public story.
- We made axis labels larger and more explicit so the plots explain themselves.
- We moved legends away from titles and increased chart margins so the figures read cleanly.
- We replaced dense methods paragraphs with a short NHANES explanation and four large takeaway bullets.

## Lessons

- Progressive disclosure is essential. Keep the default screen focused on the chart, then let people opt into details.
- One strong message per section works better than several small messages shown at once.
- If a chart needs a permanent interpretation box, the chart or axis labels are probably not doing enough work.
- Visible controls carry a cognitive cost. Every control should earn its place in the default view.
- Large type and clear labels reduce explanation needs more effectively than extra prose.
- Public-facing scientific UI should not look empty, but it also should not ask the viewer to parse everything at once.
- Caveats matter, but they should stay accessible without dominating the first read.

## Working Rule

When adding a new public-facing feature, start with the smallest readable default state.
Then ask:

- Can this explanation stay hidden until clicked?
- Can the chart label do this job instead?
- Does this control change the main story, or is it just another option?
- Is the first screen still understandable in under five seconds?
