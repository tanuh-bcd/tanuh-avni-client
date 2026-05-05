import {assert} from "chai";
import IndividualSearchCriteria from "../../../src/service/query/IndividualSearchCriteria";

describe("IndividualSearchCriteriaTest", () => {
    describe("subject answer filters", () => {
        it("includes allowedSubjectUUIDs as an OR clause when set", () => {
            const criteria = IndividualSearchCriteria.empty();
            criteria.addAllowedSubjectUUIDsCriteria(["uuid-a", "uuid-b"]);

            const filter = criteria.getFilterCriteria();
            assert.include(filter, `uuid = "uuid-a"`);
            assert.include(filter, `uuid = "uuid-b"`);
            assert.include(filter, `uuid = "uuid-a" OR uuid = "uuid-b"`);
        });

        it("includes excludedSubjectUUIDs as an AND clause when set", () => {
            const criteria = IndividualSearchCriteria.empty();
            criteria.addExcludedSubjectUUIDsCriteria(["uuid-a", "uuid-b"]);

            const filter = criteria.getFilterCriteria();
            assert.include(filter, `uuid != "uuid-a"`);
            assert.include(filter, `uuid != "uuid-b"`);
            assert.include(filter, `uuid != "uuid-a" AND uuid != "uuid-b"`);
        });

        // allowed and excluded UUIDs are mutually exclusive by design — guarded
        // upstream in FormElementStatusBuilder.build() and the FormElementStatus
        // constructor — so we don't test a combined state here.

        it("omits allowed/excluded clauses when not set", () => {
            const criteria = IndividualSearchCriteria.empty();
            const filter = criteria.getFilterCriteria();
            assert.notInclude(filter, "uuid = ");
            assert.notInclude(filter, "uuid != ");
        });

        it("preserves excludedSubjectUUIDs across clone", () => {
            const criteria = IndividualSearchCriteria.empty();
            criteria.addExcludedSubjectUUIDsCriteria(["uuid-a"]);

            const cloned = criteria.clone();
            assert.deepEqual(cloned.excludedSubjectUUIDs, ["uuid-a"]);
            assert.include(cloned.getFilterCriteria(), `uuid != "uuid-a"`);
        });
    });
});
