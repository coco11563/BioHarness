"""ClinicalTrials.gov search tool.

This module provides tools to search and retrieve clinical trial information.

APIs used:
- ClinicalTrials.gov API v2 (https://clinicaltrials.gov/api/v2/)

Example:
    resolver = ClinicalTrialsResolver()
    results = await resolver.search_studies(condition="diabetes", intervention="metformin")
    trial = await resolver.get_study("NCT00000102")
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2"


@dataclass
class TrialOutcome:
    """Primary/secondary outcome entry."""
    type: str  # "primary" or "secondary"
    measure: str = ""
    time_frame: str = ""
    description: str = ""


@dataclass
class EligibilityCriteria:
    """Eligibility criteria block."""
    criteria_text: str = ""
    sex: str = ""  # "All", "Female", "Male"
    minimum_age: str = ""
    maximum_age: str = ""
    healthy_volunteers: Optional[bool] = None


@dataclass
class TrialSummary:
    """Search result summary."""
    nct_id: str
    title: str = ""
    status: str = ""
    phases: List[str] = field(default_factory=list)
    enrollment: Optional[int] = None
    conditions: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)


@dataclass
class TrialDetails:
    """Full study details."""
    nct_id: str
    title: str = ""
    status: str = ""
    phases: List[str] = field(default_factory=list)
    enrollment: Optional[int] = None
    start_date: str = ""
    completion_date: str = ""
    conditions: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    eligibility: Optional[EligibilityCriteria] = None
    outcomes: List[TrialOutcome] = field(default_factory=list)
    sponsor: str = ""


class ClinicalTrialsResolver:
    """ClinicalTrials.gov API v2 client.

    Thread-safe singleton pattern. Uses aiohttp for async API access.
    All operations are fail-fast (no retries, exceptions propagate).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, timeout: float = 15.0, default_page_size: int = 10):
        if self._initialized:
            return
        self.timeout = timeout
        self.default_page_size = default_page_size
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = True

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request returning JSON. Fail-fast on errors."""
        session = await self._get_session()
        url = f"{BASE_URL}{path}"
        async with session.get(url, params=params or {}) as resp:
            resp.raise_for_status()  # fail-fast
            return await resp.json()

    @staticmethod
    def _extract_interventions(arms_module: Dict[str, Any]) -> List[str]:
        """Extract intervention names from arms/interventions module."""
        interventions = []
        for item in arms_module.get("interventions", []):
            name = item.get("name", "")
            if name:
                interventions.append(name)
        return interventions

    @staticmethod
    def _extract_outcomes(outcomes_module: Dict[str, Any]) -> List[TrialOutcome]:
        """Extract primary and secondary outcomes."""
        outcomes = []

        for po in outcomes_module.get("primaryOutcomes", []):
            outcomes.append(TrialOutcome(
                type="primary",
                measure=po.get("measure", ""),
                time_frame=po.get("timeFrame", ""),
                description=po.get("description", ""),
            ))

        for so in outcomes_module.get("secondaryOutcomes", []):
            outcomes.append(TrialOutcome(
                type="secondary",
                measure=so.get("measure", ""),
                time_frame=so.get("timeFrame", ""),
                description=so.get("description", ""),
            ))

        return outcomes

    @staticmethod
    def _extract_eligibility(elig_module: Dict[str, Any]) -> EligibilityCriteria:
        """Extract eligibility criteria."""
        healthy_vol = elig_module.get("healthyVolunteers")
        if healthy_vol is not None:
            healthy_vol = str(healthy_vol).lower() == "true"

        return EligibilityCriteria(
            criteria_text=elig_module.get("eligibilityCriteria", ""),
            sex=elig_module.get("sex", ""),
            minimum_age=elig_module.get("minimumAge", ""),
            maximum_age=elig_module.get("maximumAge", ""),
            healthy_volunteers=healthy_vol,
        )

    def _parse_study_summary(self, study: Dict[str, Any]) -> TrialSummary:
        """Parse a study into TrialSummary."""
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        conditions_mod = proto.get("conditionsModule", {})
        arms = proto.get("armsInterventionsModule", {})

        enrollment = None
        enroll_info = design.get("enrollmentInfo", {})
        if enroll_info:
            enrollment = enroll_info.get("count")

        return TrialSummary(
            nct_id=ident.get("nctId", ""),
            title=ident.get("briefTitle", ""),
            status=status_mod.get("overallStatus", ""),
            phases=design.get("phases", []),
            enrollment=enrollment,
            conditions=conditions_mod.get("conditions", []),
            interventions=self._extract_interventions(arms),
        )

    def _parse_study_details(self, study: Dict[str, Any]) -> TrialDetails:
        """Parse a study into TrialDetails."""
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        conditions_mod = proto.get("conditionsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        elig = proto.get("eligibilityModule", {})
        outcomes_mod = proto.get("outcomesModule", {})
        sponsors = proto.get("sponsorCollaboratorsModule", {})

        enrollment = None
        enroll_info = design.get("enrollmentInfo", {})
        if enroll_info:
            enrollment = enroll_info.get("count")

        start_date = ""
        start_struct = status_mod.get("startDateStruct", {})
        if start_struct:
            start_date = start_struct.get("date", "")

        completion_date = ""
        comp_struct = status_mod.get("completionDateStruct", {})
        if comp_struct:
            completion_date = comp_struct.get("date", "")

        sponsor = ""
        lead_sponsor = sponsors.get("leadSponsor", {})
        if lead_sponsor:
            sponsor = lead_sponsor.get("name", "")

        return TrialDetails(
            nct_id=ident.get("nctId", ""),
            title=ident.get("briefTitle", ""),
            status=status_mod.get("overallStatus", ""),
            phases=design.get("phases", []),
            enrollment=enrollment,
            start_date=start_date,
            completion_date=completion_date,
            conditions=conditions_mod.get("conditions", []),
            interventions=self._extract_interventions(arms),
            eligibility=self._extract_eligibility(elig) if elig else None,
            outcomes=self._extract_outcomes(outcomes_mod),
            sponsor=sponsor,
        )

    async def search_studies(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        keyword: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> List[TrialSummary]:
        """Search clinical trials by condition, intervention, or keyword.

        Args:
            condition: Disease/condition to search (e.g., "diabetes")
            intervention: Drug/treatment to search (e.g., "metformin")
            keyword: General search term
            page_size: Maximum results (default 10)

        Returns:
            List of TrialSummary objects

        Raises:
            ValueError: If no search terms provided
            aiohttp.ClientError: On API errors (fail-fast)
        """
        if not any([condition, intervention, keyword]):
            raise ValueError("At least one search term (condition, intervention, or keyword) required")

        params: Dict[str, Any] = {}
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if keyword:
            params["query.term"] = keyword

        params["pageSize"] = page_size or self.default_page_size

        data = await self._get_json("/studies", params=params)
        studies = data.get("studies", [])
        return [self._parse_study_summary(s) for s in studies]

    async def get_study(self, nct_id: str) -> Optional[TrialDetails]:
        """Get full study details by NCT ID.

        Args:
            nct_id: ClinicalTrials.gov NCT ID (e.g., "NCT00000102")

        Returns:
            TrialDetails or None if not found

        Raises:
            ValueError: If nct_id is empty
            aiohttp.ClientError: On API errors (fail-fast)
        """
        if not nct_id or not nct_id.strip():
            raise ValueError("nct_id must be non-empty")

        nct_id = nct_id.strip()

        try:
            data = await self._get_json(f"/studies/{nct_id}")
            return self._parse_study_details(data)
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise

    async def get_study_eligibility(self, nct_id: str) -> Optional[EligibilityCriteria]:
        """Get study eligibility criteria.

        Args:
            nct_id: NCT ID

        Returns:
            EligibilityCriteria or None
        """
        study = await self.get_study(nct_id)
        return study.eligibility if study else None

    async def get_study_outcomes(self, nct_id: str) -> List[TrialOutcome]:
        """Get study outcomes.

        Args:
            nct_id: NCT ID

        Returns:
            List of TrialOutcome objects
        """
        study = await self.get_study(nct_id)
        return study.outcomes if study else []


# Convenience async function
async def search_trials(
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    limit: int = 10,
) -> List[TrialSummary]:
    """Convenience function to search trials."""
    resolver = ClinicalTrialsResolver()
    return await resolver.search_studies(
        condition=condition,
        intervention=intervention,
        page_size=limit,
    )
