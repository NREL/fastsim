pub(crate) mod serde_db_path {
    pub(crate) fn serialize<S>(
        db_path: &Option<crate::vehicle::database::VehicleSchema>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: ::serde::Serializer,
    {
        match db_path {
            Some(path) => {
                let raw = match path {
                    crate::vehicle::database::VehicleSchema::V1(v1) => v1.to_string(),
                };
                serializer.serialize_some(&raw)
            }
            None => serializer.serialize_none(),
        }
    }

    pub(crate) fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Option<crate::vehicle::database::VehicleSchema>, D::Error>
    where
        D: ::serde::Deserializer<'de>,
    {
        let raw = <Option<String> as ::serde::Deserialize>::deserialize(deserializer)?;
        raw.map(|path| path.parse().map_err(::serde::de::Error::custom))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
    struct DbPathWrapper {
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            with = "serde_db_path"
        )]
        db_path: Option<crate::vehicle::database::VehicleSchema>,
    }

    #[test]
    fn db_path_serde_adapter_round_trip() {
        let raw_path = "v1/fastsim-3/conv/ford/fusion/2012/base/r1";
        let expected = DbPathWrapper {
            db_path: Some(raw_path.parse().expect("valid schema path")),
        };

        let json = ::serde_json::to_string(&expected).expect("serialize wrapper");
        assert_eq!(json, format!("{{\"db_path\":\"{raw_path}\"}}"));

        let parsed: DbPathWrapper = ::serde_json::from_str(&json).expect("deserialize wrapper");
        assert_eq!(parsed, expected);
    }
}
