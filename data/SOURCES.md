# Document sources

All documents are publicly posted by the employer or agency. No PII. Retrieved 2026-10-09.

| File | Document | Issuer | Source URL |
|---|---|---|---|
| `docs/lincoln_moravian_ltd.pdf` | Certificate of Group Long-Term Disability Insurance (GL3002-LTD-CERT), Moravian University, 40 pp. | The Lincoln National Life Insurance Company | https://moravian.edu/content/long-term-disability-ltd-certificate-coverage |
| `docs/standard_uah_ltd.pdf` | Certificate, Group Long Term Disability Insurance, policy 643197, University of Alabama in Huntsville, 29 pp. | Standard Insurance Company | https://www.uah.edu/images/administrative/human-resources/ltd_certificate_of_coverage.pdf |
| `docs/securian_colorado_term_life.pdf` | Employee Group Term Life Certificate of Insurance, policy 33780, State of Colorado (Rev 1-2025), 27 pp. | Minnesota Life Insurance Company (Securian) | https://dhr.colorado.gov/sites/dhr/files/documents/State%20of%20Colorado%2033780%20Term%20Life%20Certificate%20Eff%202-1-2025.pdf |
| `docs/dol_fmla_employee_guide.pdf` | Employee's Guide to the Family and Medical Leave Act, 20 pp. | U.S. Department of Labor, Wage and Hour Division | https://www.dol.gov/sites/dolgov/files/WHD/legacy/files/employeeguide.pdf (downloaded manually; dol.gov blocks scripted downloads) |

## Notes for retrieval and evaluation
- Carriers use different words for the same concept. Lincoln says "Elimination Period" and "Pre-Existing"; The Standard says "Benefit Waiting Period" and "Preexisting". Keyword search would miss these; embeddings should not. This is worth testing in the golden set.
- Certificates summarize the group policy; each states the policy governs if they conflict. Answers should cite the certificate, not claim to state the policy.
